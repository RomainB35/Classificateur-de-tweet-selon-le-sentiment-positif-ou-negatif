# 📊 Analyse et Modélisation de Tweets — MLOps & NLP

Ce dépôt contient l'ensemble des notebooks et scripts Python utilisés pour extraire, analyser, et modéliser des données textuelles (tweets). Il s'appuie sur des techniques de traitement du langage naturel (NLP), des méthodes de classification classiques et avancées, ainsi qu'une infrastructure MLOps avec **MLflow**.

---

## 🗂 Structure du répertoire

├── fonctions_analyse.py
├── fonctions_extraction.py
├── fonctions_modelisation.py
├── fonctions_modelisation_avancee.py
├── Modelisation-simple-extraction-embeddings.ipynb
├── Modelisation-simple-analyse.ipynb
├── Modelisation-simple-classification.ipynb
├── Modelisation-avancee.ipynb



---

## 🧠 Objectifs

- Extraire des **features numériques** à partir de tweets.
- Visualiser des **embeddings textuels** en 2D et 3D.
- Appliquer des **classificateurs simples** sur des représentations réduites.
- Mettre en œuvre une **pipeline avancée** de modélisation avec suivi MLOps via **MLflow**.

---

## 📝 Notebooks

### 1. `Modelisation-simple-extraction-embeddings.ipynb`
- **But** : Extraction de features numériques à partir de données textuelles.
- **Méthodes** : Universal Sentence Encoder (USE), TF-IDF, etc.
- **Données** : Tweets textuels bruts.
- **Sorties** : Matrices d'embeddings exploitables pour la modélisation.

### 2. `Modelisation-simple-analyse.ipynb`
- **But** : Représenter les embeddings en 2D/3D via PCA
- **Objectif** : Étudier la faisabilité d'une classification supervisée.

### 3. `Modelisation-simple-classification.ipynb`
- **But** : Appliquer des modèles simples de classification.
- **Modèles** : Régression logistique, Random Forest, XGBoost.
- **Entrée** : Embeddings extraits et réduits.

### 4. `Modelisation-avancee.ipynb`
- **But** : Entraîner des modèles d’embedding avancés sur un serveur **MLflow** local.
- **Modèles** :
  - Word2Vec
  - GloVe
  - BERT (transformers)
- **Outils** :
  - MLflow pour le suivi des expériences (tracking, logging, stockage)
  - Intégration MLOps pour un cycle de vie reproductible des modèles

---

## 🛠 Scripts Python

| Fichier | Rôle |
|--------|------|
| `fonctions_extraction.py` | Fonctions pour l’extraction de features (embeddings, preprocessing). |
| `fonctions_analyse.py` | Fonctions pour l’analyse exploratoire et la visualisation des embeddings. |
| `fonctions_modelisation.py` | Fonctions pour les modèles de classification classiques. |
| `fonctions_modelisation_avancee.py` | Fonctions dédiées à la modélisation avancée et à l’intégration avec MLflow. |

---

## 🚀 Dépendances

Toutes les dépendances sont dans le fichier requirements.txt.

---

