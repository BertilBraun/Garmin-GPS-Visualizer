gcloud init
gcloud config set project garmin-gps-visualizer
gcloud services enable   run.googleapis.com   cloudbuild.googleapis.com   artifactregistry.googleapis.com   firestore.googleapis.com   storage.googleapis.com

# NOTE: Create Firestore DB in Cloud UI


set BUCKET_NAME=garmin-gpx-garmin-gps-visualizer
gcloud storage buckets create gs://%BUCKET_NAME% --location=EU --uniform-bucket-level-access


set BUCKET_NAME=garmin-gpx-garmin-gps-visualizer
gcloud run deploy garmin-gps-visualizer --source . --region europe-west1 --allow-unauthenticated --set-env-vars BUCKET_NAME=%BUCKET_NAME%



## DEBUG

set BUCKET_NAME=garmin-gpx-garmin-gps-visualizer
uvicorn app:APP --reload --log-level debug
