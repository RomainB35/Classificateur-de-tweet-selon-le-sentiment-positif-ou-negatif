# 1. Installer le SDK Google Cloud
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# 2. Init et login
gcloud init
gcloud auth login

# 3. Sélection du projet
gcloud config set project YOUR-PROJECT-ID

# 4. Activer les services requis
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

