# app.py
from __future__ import annotations

import logging
import os
import re
import time
import math
import traceback
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

SYNC_HTTP_TIMEOUT_S = int(os.environ.get('SYNC_HTTP_TIMEOUT_S', '300'))
SYNC_GPX_PARALLELISM = int(os.environ.get('SYNC_GPX_PARALLELISM', '10'))

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
    try:
        api.garth.configure(timeout=SYNC_HTTP_TIMEOUT_S)
    except Exception:
        pass
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


_garmin_threadlocal = threading.local()


def _thread_garmin_api(tokenstore: str) -> Garmin:
    api = getattr(_garmin_threadlocal, 'api', None)
    if api is not None and getattr(_garmin_threadlocal, 'tokenstore', None) == tokenstore:
        return api

    api = Garmin()
    api.garth.loads(tokenstore)
    try:
        api.garth.configure(timeout=SYNC_HTTP_TIMEOUT_S)
    except Exception:
        pass

    _garmin_threadlocal.api = api
    _garmin_threadlocal.tokenstore = tokenstore
    return api


def _download_gpx_and_mean(tokenstore: str, activity_id: int) -> Tuple[int, bytes, Optional[Tuple[float, float]]]:
    api = _thread_garmin_api(tokenstore)
    for attempt in range(4):
        try:
            data = download_gpx(api, activity_id)
            pts = parse_gpx_points_from_bytes(data)
            mp = mean_point(pts)
            return activity_id, data, mp
        except GarminConnectTooManyRequestsError:
            if attempt >= 3:
                raise
            time.sleep(10 * (attempt + 1) + random.random() * 2)


# -------------------------
# Activity parsing helpers
# -------------------------


def activity_type_key(activity: Dict[str, Any], email: str) -> str:
    t = activity.get('activityType') or {}
    type_key = (t.get('typeKey') or t.get('typeName') or activity.get('activityTypeName') or '').strip().lower()

    if type_key == 'other' and email == 'bertil.braun.private@gmail.com':
        return 'windsurfing_v2'
    return re.sub(r'[^a-z0-9_]+', '', type_key) or 'unknown'


def activity_display_name(activity: Dict[str, Any]) -> str:
    return str(
        activity.get('activityName') or activity.get('activityTitle') or activity.get('activityTypeName') or 'Activity'
    )


def activity_has_polyline(activity: Dict[str, Any]) -> bool:
    v = activity.get('hasPolyline')
    return bool(v) if v is not None else False


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
def sync_activities(email: str, password: str, spot_radius_m: float) -> SyncResponse:
    user_id = user_id_from_email(email)
    col = activities_collection(user_id)

    api = get_garmin_api(email, password)
    tokenstore = api.garth.dumps()
    acts = list_activities(api)

    existing_docs = {int(d.id): (d.to_dict() or {}) for d in col.stream()}

    download_ids: List[int] = []
    for a in acts:
        aid = get_activity_id(a)
        if aid is None:
            continue
        if aid in existing_docs:
            continue
        if activity_has_polyline(a):
            download_ids.append(aid)

    downloads: Dict[int, Tuple[bytes, Optional[Tuple[float, float]]]] = {}
    if download_ids:
        with ThreadPoolExecutor(max_workers=max(1, SYNC_GPX_PARALLELISM)) as ex:
            futs = {ex.submit(_download_gpx_and_mean, tokenstore, aid): aid for aid in download_ids}
            for fut in as_completed(futs):
                aid, data, mp = fut.result()
                downloads[aid] = (data, mp)

    new_uploaded = 0
    created_docs = 0

    for a in sorted(acts, key=lambda x: int(get_activity_id(x) or 0)):
        aid = get_activity_id(a)
        if aid is None:
            continue

        doc_ref = col.document(str(aid))
        if aid in existing_docs:
            existing = existing_docs[aid]
            updates: Dict[str, Any] = {}
            if not existing.get('typeKey'):
                updates['typeKey'] = activity_type_key(a, email)
            if not existing.get('name'):
                updates['name'] = activity_display_name(a)
            if not existing.get('startTime'):
                updates['startTime'] = parse_start_time(a)
            if not existing.get('connectUrl'):
                updates['connectUrl'] = connect_url(aid)
            if updates:
                doc_ref.set(updates, merge=True)
            continue

        blob_name = None
        mp = None
        if activity_has_polyline(a):
            got = downloads.get(aid)
            if got is not None:
                data, mp = got
                blob = bucket.blob(gpx_object_path(user_id, aid))
                blob.upload_from_string(data, content_type='application/gpx+xml')
                blob_name = blob.name
                new_uploaded += 1

        meta = {
            'activityId': aid,
            'name': activity_display_name(a),
            'startTime': parse_start_time(a),
            'typeKey': activity_type_key(a, email),
            'connectUrl': connect_url(aid),
            'gpxObject': blob_name,
            'meanLat': mp[0] if mp else None,
            'meanLon': mp[1] if mp else None,
            'createdAt': firestore.SERVER_TIMESTAMP,
        }
        doc_ref.set(meta)
        created_docs += 1

    last_sync = datetime.utcnow().isoformat() + 'Z'
    fs.collection('users').document(user_id).set({'lastSync': last_sync}, merge=True)

    indexed = len(existing_docs) + created_docs

    return SyncResponse(user_id=user_id, indexed=indexed, new_gpx_uploaded=new_uploaded, last_sync=last_sync)


# -------------------------
# API endpoints
# -------------------------
@APP.post('/api/sync', response_model=SyncResponse)
def api_sync(body: SyncRequest):
    try:
        return sync_activities(body.email, body.password, body.spot_radius_m)

    except (GarminConnectAuthenticationError, GarminConnectConnectionError) as e:
        raise HTTPException(status_code=401, detail=f'Garmin login failed: {e}')

    except GarminConnectTooManyRequestsError:
        raise HTTPException(status_code=429, detail='Rate limited by Garmin. Retry later.')

    except Exception as e:
        tb = traceback.format_exc()
        log.error('Unexpected sync error: %s\n%s', e, tb)
        raise HTTPException(status_code=500, detail={'error': str(e), 'traceback': tb})


@APP.get('/api/{user_id}/types')
def list_user_activity_types(user_id: str):
    col = activities_collection(user_id)
    docs = list(col.stream())
    if not docs:
        raise HTTPException(status_code=404, detail='No activities for this user_id (sync first).')

    counts: Dict[str, int] = {}
    for d in docs:
        x = d.to_dict() or {}
        k = str(x.get('typeKey') or 'unknown').strip().lower() or 'unknown'
        counts[k] = counts.get(k, 0) + 1

    items = [{'typeKey': k, 'count': v} for (k, v) in counts.items()]
    items.sort(key=lambda m: (-int(m['count']), str(m['typeKey'])))
    return JSONResponse(items)


@APP.get('/api/{user_id}/activities')
def list_user_activities(user_id: str, type: Optional[str] = None):
    col = activities_collection(user_id)
    docs = list(col.stream())
    if not docs:
        raise HTTPException(status_code=404, detail='No activities for this user_id (sync first).')

    type_key = (type or '').strip().lower()
    items: List[Dict[str, Any]] = []
    for d in docs:
        x = d.to_dict() or {}
        if type_key and str(x.get('typeKey') or '').strip().lower() != type_key:
            continue
        items.append(
            {
                'activityId': int(x['activityId']),
                'name': x.get('name') or 'Activity',
                'typeKey': x.get('typeKey') or 'unknown',
                'startTime': x.get('startTime'),
                'connectUrl': x.get('connectUrl') or connect_url(int(x['activityId'])),
                'meanLat': x.get('meanLat'),
                'meanLon': x.get('meanLon'),
            }
        )

    items.sort(key=lambda m: (m.get('startTime') or ''), reverse=True)
    return JSONResponse(items)


@APP.get('/api/{user_id}/spots')
def spots(user_id: str, radius_m: float = 1200.0, type: Optional[str] = None):
    """
    Spots are computed on the fly from per-activity mean points and clustered by radius.
    """
    col = activities_collection(user_id)
    docs = list(col.stream())
    if not docs:
        raise HTTPException(status_code=404, detail='No activities for this user_id (sync first).')

    type_key = (type or '').strip().lower()
    pts: List[Tuple[int, Tuple[float, float]]] = []
    for d in docs:
        x = d.to_dict() or {}
        if type_key and str(x.get('typeKey') or '').strip().lower() != type_key:
            continue
        if x.get('meanLat') is None or x.get('meanLon') is None:
            continue
        pts.append((int(x['activityId']), (float(x['meanLat']), float(x['meanLon']))))

    out = cluster_spots(pts, radius_m=radius_m)
    return JSONResponse(out)


@APP.get('/api/{user_id}/spot/{spot_id}/activities')
def spot_activities(user_id: str, spot_id: int, radius_m: float = 1200.0, type: Optional[str] = None):
    # recompute spots and return activities for the selected one
    col = activities_collection(user_id)
    docs = list(col.stream())
    if not docs:
        raise HTTPException(status_code=404, detail='No activities for this user_id (sync first).')

    items: List[Dict[str, Any]] = []
    pts: List[Tuple[int, Tuple[float, float]]] = []
    by_id: Dict[int, Dict[str, Any]] = {}
    type_key = (type or '').strip().lower()

    for d in docs:
        x = d.to_dict() or {}
        if type_key and str(x.get('typeKey') or '').strip().lower() != type_key:
            continue
        aid = int(x['activityId'])
        by_id[aid] = {
            'activityId': aid,
            'name': x.get('name') or 'Activity',
            'typeKey': x.get('typeKey') or 'unknown',
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
        raise HTTPException(status_code=404, detail='No GPS track available for this activity.')

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
