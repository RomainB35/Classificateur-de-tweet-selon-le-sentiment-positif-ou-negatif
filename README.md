# 🧠 Classification de tweets selon le sentiment (positif ou négatif)

Ce projet a pour but principal  de fournir un **serveur d'inférence** permettant de prédire le **sentiment (positif ou négatif) d'un tweet (en anglais)** à partir de plusieurs modèles d’apprentissage automatique, allant d’approches classiques à des modèles avancés basés sur BERT.

L'entraînement des modèles est réalisé à partir d’un jeu de données de tweets anglais annoté contenant **1 600 000 tweets** (0 = Négatif, 4 = Positif), disponible sur Kaggle :  
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
├── dockerfiles/                        # Fichiers utilisés pour construire les images docker et définir les dépendances utilisées ainsi que les applications Fastapi et Streamlit
├── deploy_on_cloud/                    # Scripts utilisés pour déployer le serveur d'inférence sur google cloud

```
## 🐳 Installation de Docker

Pour installer Docker sur votre machine, suivre la documentation officielle :  
👉 https://docs.docker.com/engine/install/

Une fois installé, vérifier que Docker fonctionne :

```bash
docker --version
```

---
## Pour build les images Docker

```bash
 docker build -t bert-fastapi-cpu -f dockerfiles/bert-fastapi-cpu/Dockerfile .
 docker build -t bert-fastapi-streamlit-azure -f bert-fastapi-streamlit-azure/Dockerfile .
```

## 🚀 Exploitation de l'image Docker publique

L'image docker légère qui contient uniquement le serveur d'inférence exposé via API est disponible publiquement et prête à l'emploi.

Cette image est aussi déployée sur google cloud.

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

# 🧪 Tester l'API BERT FastAPI déployée sur Cloud Run

Vous pouvez tester les différents endpoints de l'API avec `curl`. Assurez-vous d'avoir `jq` installé pour un affichage lisible des réponses JSON.

La première requête prend plus de temps à s'éxécuter si le container n'est pas déjà démarré.

La documentation de l'API est disponible ici 👉 https://bert-fastapi-service-70236624058.europe-west1.run.app/docs .

---

## 🔍 1. Tester le endpoint racine (`/`)

Ce endpoint permet simplement de vérifier que l'API est opérationnelle.

```bash
curl -X GET https://bert-fastapi-service-70236624058.europe-west1.run.app/ | jq
```

Réponse attendue :
```bash
{
  "message": "Service BERT FastAPI is running."
}
```

## 💬 2. Prédiction sur un tweet unique (/predict)

Ce endpoint renvoie le sentiment d'un tweet (positif ou négatif).

```bash
curl -X POST https://bert-fastapi-service-70236624058.europe-west1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love Cloud Run!"}' | jq
```

Réponse attendue :
```bash
{
  "tweet": "I love Cloud Run!",
  "prediction": 4,
  "confidence": 0.89,
  "sentiment": "Tweet positif"
}
```

## 📚 3. Prédiction sur plusieurs tweets (/predict_batch)

Ce endpoint accepte une liste de textes et renvoie une prédiction pour chacun.

```bash
curl -X POST https://bert-fastapi-service-70236624058.europe-west1.run.app/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I love Cloud Run!",
      "This is terrible.",
      "BERT is awesome!"
    ]
  }' | jq
  ```

Réponse attendue :
```bash
[
  {
    "tweet": "I love Cloud Run!",
    "prediction": 4,
    "confidence": 0.89,
    "sentiment": "Tweet positif"
  },
  {
    "tweet": "This is terrible.",
    "prediction": 0,
    "confidence": 0.92,
    "sentiment": "Tweet négatif"
  },
  {
    "tweet": "BERT is awesome!",
    "prediction": 4,
    "confidence": 0.93,
    "sentiment": "Tweet positif"
  }
]
```
ℹ️ Remarques

    Tous les endpoints acceptent et renvoient du JSON.

    Le modèle renvoie un score de prediction entre 0 (négatif) et 4 (positif).

    La clé confidence correspond à la probabilité associée à la prédiction.








