# 🧠 Classification de tweets selon le sentiment (positif ou négatif)

Ce projet a pour but de fournir un **serveur d'inférence** permettant de prédire le **sentiment d'un tweet** à partir de plusieurs modèles d’apprentissage automatique, allant d’approches classiques à des modèles avancés basés sur BERT.

L'entraînement des modèles est réalisé à partir d’un jeu de données annoté contenant **1 600 000 tweets** (0 = Négatif, 4 = Positif), disponible sur Kaggle :  
🔗 https://www.kaggle.com/datasets/kazanova/sentiment140

L'ensemble du projet est packagé sous forme d’**images Docker** pour garantir la portabilité. Il intègre :
- Le **suivi des expérimentations** via **MLFlow**
- Une **API d'inférence** via **FastAPI**
- Une **interface utilisateur** via **Streamlit**
- Un système de **feedback utilisateur** connecté à **Azure Application Insights** pour le monitoring et la traçabilité.

---

## 📁 Structure du projet

```bash
Classificateur-de-tweet-selon-le-sentiment-positif-ou-negatif/
│
├── notebooks/                          # Modélisation + tracking des expérimentations via MLFlow
├── mlflow-server/                      # Image Docker contenant un serveur MLFlow
├── bert-fastapi-cpu/                   # Image Docker avec serveur d'inférence BERT (FastAPI) optimisé pour CPU
├── bert-fastapi-streamlit-azure/      # Image Docker avec interface web (Streamlit) + API FastAPI + logs Azure Insights
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

