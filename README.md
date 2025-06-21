# 🧠 Classification de tweets selon le sentiment (positif ou négatif)

Ce projet a pour but principal  de fournir un **serveur d'inférence** permettant de prédire le **sentiment (positif ou négatif) d'un tweet** à partir de plusieurs modèles d’apprentissage automatique, allant d’approches classiques à des modèles avancés basés sur BERT.

L'entraînement des modèles est réalisé à partir d’un jeu de données annoté contenant **1 600 000 tweets** (0 = Négatif, 4 = Positif), disponible sur Kaggle :  
🔗 https://www.kaggle.com/datasets/kazanova/sentiment140

Pour l'entrainement, le stockage et la mesure des performances des modèles, j'utilise un serveur MLflow local, pour l'installation voir 👉 https://www.mlflow.org/docs/latest/ml/tracking/quickstart.

Pour faciliter la portabilité du serveur d'inférence, celui-ci sera contenu dans une **image docker**.

Deux images dockers seront construites:

 - 1. Une image légère utilisant un modèle BERT entrainé sur les données via MLflow pour l'inférence, exposant une API (FastApi) et conçue pour tourner sur du CPU.
 - 2. Une image plus lourde qui inclue en plus une interface web interactive (Streamlit) qui permet à un utilisateur de saisir un tweet, d'obtenir une prédiction et d'envoyer un feedback qui sera remonté sur une instance Azure Application Insights pour superviser les performances. 


---

## 📁 Structure du projet

```bash
Classificateur-de-tweet-selon-le-sentiment-positif-ou-negatif/
│
├── notebooks/                          # Modélisation + tracking des expérimentations via MLFlow
├── saved_model/                        # Artefacts extraits de l'entrainement du modèle BERT via le serveur MLflow qui seront utilisés pour construire le serveur d'inférence
├── dockerfiles/                        # Fichiers utilisés pour définir les dépendances utilisées pour les images docker ainsi que les applications Fastapi et Streamlit
├── bert-fastapi-cpu/                   # Construction d'une image Docker avec serveur d'inférence BERT (FastAPI) optimisé pour CPU
├── bert-fastapi-streamlit-azure/       # Construction d'une image Docker avec interface web (Streamlit) + API FastAPI + logs Azure Insights
├── scripts/                            # Scripts d'entraînement, évaluation, export
```

---

## 🐳 Installation de Docker

Pour installer Docker sur votre machine, suivre la documentation officielle :  
👉 https://docs.docker.com/engine/install/

Une fois installé, vérifier que Docker fonctionne :

```bash
docker --version
```

---

## 🚀 Exploitation des images Docker publiques

Les images Docker du **serveur MLFlow** et du **serveur d'inférence BERT-FastAPI-CPU** sont disponibles publiquement et prêtes à l'emploi.

### 🔹 Télécharger l’image d'inférence

```bash
docker pull ghcr.io/romainb35/bert-fastapi-cpu:latest
```

### 🔹 Lancer l’API localement

```bash
docker run -d -p 8000:8000 ghcr.io/romainb35/bert-fastapi-cpu:latest
```

Cela lance un conteneur en arrière-plan, accessible à l’adresse :  
📍 http://localhost:8000/docs (documentation Swagger de l’API)

---

