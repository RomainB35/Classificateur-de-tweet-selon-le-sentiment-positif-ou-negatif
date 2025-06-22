#!/bin/bash

# === CONFIG ===
PROJECT_ID="YOUR-PROJECT-ID"
REGION="europe-west1"
REPO_NAME="bert-repo"
SERVICE_NAME="bert-fastapi-service"
GHCR_IMAGE="ghcr.io/romainb35/bert-fastapi-cpu:latest"
GCP_IMAGE="europe-west1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/bert-fastapi-cpu:latest"

# === SETUP ===
echo "🔧 Configuration du projet"
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# === CREATION DU DEPOT SI BESOIN ===
echo "📦 Vérification du dépôt Artifact Registry..."
gcloud artifacts repositories describe $REPO_NAME --location=$REGION 2>/dev/null \
  || gcloud artifacts repositories create $REPO_NAME \
       --repository-format=docker \
       --location=$REGION \
       --description="Dépôt Docker pour BERT FastAPI"

# === AUTH DOCKER ===
echo "🔐 Configuration Docker pour Artifact Registry..."
gcloud auth configure-docker $REGION-docker.pkg.dev

# === PULL ET PUSH DE L’IMAGE ===
echo "⬇️ Pull depuis GHCR..."
docker pull $GHCR_IMAGE

echo "🔁 Tag vers Google Artifact Registry..."
docker tag $GHCR_IMAGE $GCP_IMAGE

echo "⬆️ Push vers Artifact Registry..."
docker push $GCP_IMAGE

# === DEPLOIEMENT CLOUD RUN ===
echo "🚀 Déploiement sur Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image=$GCP_IMAGE \
  --platform=managed \
  --region=$REGION \
  --port=8000 \
  --memory=2Gi \
  --timeout=600 \
  --allow-unauthenticated \
  --project=$PROJECT_ID

echo "✅ Terminé !"
echo "🌐 URL de service :"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"

