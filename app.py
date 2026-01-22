# app.py
from __future__ import annotations

import logging
import os
import re
import time
import math
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gpxpy
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from google.cloud import firestore
from google.cloud import storage

APP = FastAPI()

# Serve frontend
APP.mount('/web', StaticFiles(directory='web', html=True), name='web')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('garmin_spots')

BUCKET_NAME = os.environ.get('BUCKET_NAME')
if not BUCKET_NAME:
    raise RuntimeError('Missing env var BUCKET_NAME')

fs = firestore.Client()
gcs = storage.Client()
bucket = gcs.bucket(BUCKET_NAME)


@APP.get('/')
def root():
    return FileResponse('web/index.html')


# -------------------------
# Models
# -------------------------
class SyncRequest(BaseModel):
    email: str
    password: str
    spot_radius_m: float = 1200.0


class SyncResponse(BaseModel):
    user_id: str
    indexed: int
    new_gpx_uploaded: int
    last_sync: str


# -------------------------
# Helpers
# -------------------------
def user_id_from_email(email: str) -> str:
    """
    Derive from local-part only:
      - take part before '@'
      - keep only alphanumeric
      - lowercase
    Example: "foo.bar+1@gmail.com" -> "foobar1"
    """
    local = email.split('@', 1)[0]
    return re.sub(r'[^A-Za-z0-9]+', '', local).lower()


def connect_url(activity_id: int) -> str:
    return f'https://connect.garmin.com/modern/activity/{activity_id}'


def activities_collection(user_id: str):
    return fs.collection('users').document(user_id).collection('activities')


def gpx_object_path(user_id: str, activity_id: int) -> str:
    return f'gpx/{user_id}/{activity_id}.gpx'


# -------------------------
# Garmin helpers
# -------------------------
def get_garmin_api(email: str, password: str) -> Garmin:
    api = Garmin(email, password)
    api.login()
    return api


def list_activities(api: Garmin) -> List[Dict[str, Any]]:
    start, limit = 0, 100
    out: List[Dict[str, Any]] = []
    while True:
        page = api.get_activities(start, limit)
        if not page:
            break
        out.extend(page)
        if len(page) < limit:
            break
        start += limit
        time.sleep(0.2)
    return out


def download_gpx(api: Garmin, activity_id: int) -> bytes:
    return api.download_activity(str(activity_id), dl_fmt=Garmin.ActivityDownloadFormat.GPX)


# -------------------------
# Activity parsing helpers
# -------------------------
def is_windsurf(activity: Dict[str, Any]) -> bool:
    t = activity.get('activityType') or {}
    type_key = (t.get('typeKey') or t.get('typeName') or activity.get('activityTypeName') or '').lower()
    name = (activity.get('activityName') or activity.get('activityTitle') or '').lower()
    return ('windsurf' in type_key) or ('windsurf' in name)


def get_activity_id(activity: Dict[str, Any]) -> Optional[int]:
    for k in ('activityId', 'activity_id', 'id'):
        v = activity.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def parse_start_time(activity: Dict[str, Any]) -> Optional[str]:
    for k in ('startTimeLocal', 'startTimeGMT', 'startTime'):
        v = activity.get(k)
        if v:
            return str(v)
    return None


# -------------------------
# Geo helpers
# -------------------------
def parse_gpx_points_from_bytes(data: bytes) -> List[Tuple[float, float]]:
    gpx = gpxpy.parse(data.decode('utf-8', errors='ignore'))
    pts: List[Tuple[float, float]] = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                if p.latitude is not None and p.longitude is not None:
                    pts.append((float(p.latitude), float(p.longitude)))
    return pts


def mean_point(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    s_lat = 0.0
    s_lon = 0.0
    n = 0
    for lat, lon in points:
        s_lat += lat
        s_lon += lon
        n += 1
    return (s_lat / n, s_lon / n)


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def cluster_spots(points: List[Tuple[int, Tuple[float, float]]], radius_m: float) -> List[Dict[str, Any]]:
    """
    points: list of (activityId, (meanLat, meanLon))
    greedy clustering in insertion order
    """
    spots: List[Dict[str, Any]] = []

    def assign(lat: float, lon: float, aid: int) -> None:
        for s in spots:
            if haversine_m((s['lat'], s['lon']), (lat, lon)) <= radius_m:
                s['activityIds'].append(aid)
                # keep spot center as mean of assigned means (stable-ish)
                n = s['_n'] + 1
                s['lat'] = (s['lat'] * s['_n'] + lat) / n
                s['lon'] = (s['lon'] * s['_n'] + lon) / n
                s['_n'] = n
                return
        spots.append({'id': len(spots), 'lat': lat, 'lon': lon, 'activityIds': [aid], '_n': 1})

    for aid, (lat, lon) in points:
        assign(lat, lon, aid)

    for s in spots:
        s.pop('_n', None)
    return spots


def to_linestring_geojson(points: List[Tuple[float, float]]) -> Dict[str, Any]:
    return {'type': 'LineString', 'coordinates': [[lon, lat] for (lat, lon) in points]}


# -------------------------
# Sync pipeline
# -------------------------
def sync_windsurf(email: str, password: str, spot_radius_m: float) -> SyncResponse:
    user_id = user_id_from_email(email)
    col = activities_collection(user_id)

    api = get_garmin_api(email, password)
    acts = list_activities(api)

    windsurf: List[Tuple[int, Dict[str, Any]]] = []
    for a in acts:
        if not is_windsurf(a):
            continue
        aid = get_activity_id(a)
        if aid is None:
            continue
        windsurf.append((aid, a))

    new_uploaded = 0

    for aid, a in sorted(windsurf, key=lambda x: x[0]):
        doc_ref = col.document(str(aid))
        if doc_ref.get().exists:
            continue

        try:
            data = download_gpx(api, aid)
        except GarminConnectTooManyRequestsError:
            time.sleep(30)
            data = download_gpx(api, aid)

        pts = parse_gpx_points_from_bytes(data)
        mp = mean_point(pts)

        blob = bucket.blob(gpx_object_path(user_id, aid))
        blob.upload_from_string(data, content_type='application/gpx+xml')

        meta = {
            'activityId': aid,
            'name': a.get('activityName') or a.get('activityTitle') or 'Windsurfing',
            'startTime': parse_start_time(a),
            'connectUrl': connect_url(aid),
            'gpxObject': blob.name,
            'meanLat': mp[0] if mp else None,
            'meanLon': mp[1] if mp else None,
            'createdAt': firestore.SERVER_TIMESTAMP,
        }
        doc_ref.set(meta)

        new_uploaded += 1
        time.sleep(0.3)

    last_sync = datetime.utcnow().isoformat() + 'Z'
    fs.collection('users').document(user_id).set({'lastSync': last_sync}, merge=True)

    indexed = len(list(col.stream()))

    return SyncResponse(user_id=user_id, indexed=indexed, new_gpx_uploaded=new_uploaded, last_sync=last_sync)


# -------------------------
# API endpoints
# -------------------------
@APP.post('/api/sync', response_model=SyncResponse)
def api_sync(body: SyncRequest):
    try:
        return sync_windsurf(body.email, body.password, body.spot_radius_m)

    except (GarminConnectAuthenticationError, GarminConnectConnectionError) as e:
        raise HTTPException(status_code=401, detail=f'Garmin login failed: {e}')

    except GarminConnectTooManyRequestsError:
        raise HTTPException(status_code=429, detail='Rate limited by Garmin. Retry later.')

    except Exception as e:
        tb = traceback.format_exc()
        log.error('Unexpected sync error: %s\n%s', e, tb)
        raise HTTPException(status_code=500, detail={'error': str(e), 'traceback': tb})


@APP.get('/api/{user_id}/activities')
def list_user_activities(user_id: str):
    col = activities_collection(user_id)
    docs = list(col.stream())
    if not docs:
        raise HTTPException(status_code=404, detail='No activities for this user_id (sync first).')

    items: List[Dict[str, Any]] = []
    for d in docs:
        x = d.to_dict() or {}
        items.append(
            {
                'activityId': int(x['activityId']),
                'name': x.get('name') or 'Windsurfing',
                'startTime': x.get('startTime'),
                'connectUrl': x.get('connectUrl') or connect_url(int(x['activityId'])),
                'meanLat': x.get('meanLat'),
                'meanLon': x.get('meanLon'),
            }
        )

    items.sort(key=lambda m: (m.get('startTime') or ''), reverse=True)
    return JSONResponse(items)


@APP.get('/api/{user_id}/spots')
def spots(user_id: str, radius_m: float = 1200.0):
    """
    Spots are computed on the fly from per-activity mean points and clustered by radius.
    """
    col = activities_collection(user_id)
    docs = list(col.stream())
    if not docs:
        raise HTTPException(status_code=404, detail='No activities for this user_id (sync first).')

    pts: List[Tuple[int, Tuple[float, float]]] = []
    for d in docs:
        x = d.to_dict() or {}
        if x.get('meanLat') is None or x.get('meanLon') is None:
            continue
        pts.append((int(x['activityId']), (float(x['meanLat']), float(x['meanLon']))))

    out = cluster_spots(pts, radius_m=radius_m)
    return JSONResponse(out)


@APP.get('/api/{user_id}/spot/{spot_id}/activities')
def spot_activities(user_id: str, spot_id: int, radius_m: float = 1200.0):
    # recompute spots and return activities for the selected one
    col = activities_collection(user_id)
    docs = list(col.stream())
    if not docs:
        raise HTTPException(status_code=404, detail='No activities for this user_id (sync first).')

    items: List[Dict[str, Any]] = []
    pts: List[Tuple[int, Tuple[float, float]]] = []
    by_id: Dict[int, Dict[str, Any]] = {}

    for d in docs:
        x = d.to_dict() or {}
        aid = int(x['activityId'])
        by_id[aid] = {
            'activityId': aid,
            'name': x.get('name') or 'Windsurfing',
            'startTime': x.get('startTime'),
            'connectUrl': x.get('connectUrl') or connect_url(aid),
        }
        if x.get('meanLat') is not None and x.get('meanLon') is not None:
            pts.append((aid, (float(x['meanLat']), float(x['meanLon']))))

    spots_list = cluster_spots(pts, radius_m=radius_m)
    spot = next((s for s in spots_list if s['id'] == spot_id), None)
    if not spot:
        raise HTTPException(status_code=404, detail='Spot not found.')

    aids = set(spot['activityIds'])
    items = [by_id[aid] for aid in aids if aid in by_id]
    items.sort(key=lambda m: (m.get('startTime') or ''), reverse=True)
    return JSONResponse(items)


@APP.get('/api/{user_id}/activity/{activity_id}/geojson')
def activity_geojson(user_id: str, activity_id: int):
    doc = activities_collection(user_id).document(str(activity_id)).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail='Activity not found for this user_id.')

    meta = doc.to_dict() or {}
    obj = meta.get('gpxObject')
    if not obj:
        raise HTTPException(status_code=500, detail='Missing gpxObject in metadata.')

    blob = bucket.blob(obj)
    if not blob.exists():
        raise HTTPException(status_code=500, detail='GPX object missing in storage.')

    data = blob.download_as_bytes()
    pts = parse_gpx_points_from_bytes(data)

    feature = {
        'type': 'Feature',
        'properties': {
            'activityId': activity_id,
            'name': meta.get('name'),
            'startTime': meta.get('startTime'),
            'connectUrl': meta.get('connectUrl') or connect_url(activity_id),
        },
        'geometry': to_linestring_geojson(pts),
    }

    resp = JSONResponse(feature)
    resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    resp.headers['ETag'] = f'"{user_id}:{activity_id}"'
    return resp


@APP.get('/api/{user_id}/activity/{activity_id}/open')
def open_activity(user_id: str, activity_id: int):
    return RedirectResponse(connect_url(activity_id))
