# 🚀 Déploiement d'une image Docker sur Google Cloud avec Cloud Run

Ce répertoire contient le nécessaire permet de déployer automatiquement l'image Docker `ghcr.io/romainb35/bert-fastapi-cpu:latest` (hébergée sur GitHub Container Registry) sur **Google Cloud Run**, en utilisant deux scripts :

- `setup_gcloud.sh` : pour configurer votre environnement Google Cloud.
- `deploy_cloud_run.sh` : pour transférer l’image Docker vers Google Artifact Registry et déployer le service sur Cloud Run.

## 📁 Arborescence
deploy_on_cloud/
├── setup_gcloud.sh # Script de configuration initiale GCP
└── deploy_cloud_run.sh # Script de déploiement Cloud Run


---

## 🔧 Prérequis

- Un compte Google Cloud avec un projet actif.
- Docker installé localement.

---

## 1. ⚙️ Configuration de Google Cloud

Lancez le script `setup_gcloud.sh` pour installer le SDK GCP, vous connecter et activer les services requis :

```bash
./setup_gcloud.sh
```

Suivez les instructions affichées pour :

    Installer le SDK Google Cloud.

    Vous authentifier (gcloud init, gcloud auth login).

    Choisir votre projet GCP (gcloud config set project).

    Activer les APIs nécessaires : Cloud Run et Artifact Registry.

    Remplacez YOUR-PROJECT-ID dans le script par l’identifiant de votre projet GCP.


2. 🚀 Déploiement de l’image Docker sur Cloud Run

Ensuite, exécutez le script de déploiement :

```bash
./deploy_cloud_run.sh
```
Ce script effectue automatiquement les étapes suivantes :

    Configure le projet et les services requis.

    Vérifie si le dépôt Artifact Registry (bert-repo) existe, sinon le crée.

    Authentifie Docker avec Google Artifact Registry.

    Récupère l’image depuis ghcr.io/romainb35/bert-fastapi-cpu:latest.

    La re-tagge pour GCP, puis la pousse vers Artifact Registry.

    Déploie le service sur Google Cloud Run sous le nom bert-fastapi-service.

🌐 Accès à l'API

À la fin du script, l’URL publique du service sera affichée dans la console :

🌐 URL de service :
https://bert-fastapi-service-xxxxxx.a.run.app


Ce lien permet d’accéder à l’API FastAPI déployée.

📝 Remarques

    Le déploiement se fait par défaut dans la région europe-west1.

    Le service Cloud Run est public (--allow-unauthenticated).

    L’image Docker doit exposer le port 8000 (FastAPI par défaut).


🧹 Nettoyage (optionnel)

Pour supprimer le service déployé :

```bash
gcloud run services delete bert-fastapi-service --region=europe-west1
```
Et pour supprimer le dépôt Docker :
```bash
gcloud artifacts repositories delete bert-repo --location=europe-west1
```