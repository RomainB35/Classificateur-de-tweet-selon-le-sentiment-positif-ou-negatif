import os
import re
import json
import mlflow
import mlflow.pyfunc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from mlflow.models.signature import infer_signature

# --- Model Definition ---
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- Load GloVe embeddings ---
def load_glove_embeddings(embedding_path, embedding_dim=300):
    embeddings_index = {}
    with open(embedding_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

# --- Simple tokenizer ---
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# --- Text embedding ---
def embed_text(text, embeddings_index, embedding_dim=300):
    tokens = simple_tokenize(text)
    vectors = [embeddings_index.get(w, np.zeros(embedding_dim)) for w in tokens]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim)

# --- Wrapper PyFunc ---
class TextClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, embeddings_index, embedding_dim=300):
        self.model = model
        self.embeddings_index = embeddings_index
        self.embedding_dim = embedding_dim

    def _preprocess(self, text_series):
        vectors = [
            embed_text(t, self.embeddings_index, self.embedding_dim)
            for t in text_series
        ]
        return torch.tensor(vectors, dtype=torch.float32)

    def predict(self, context, model_input):
        texts = model_input["text"]
        X_embed = self._preprocess(texts)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_embed)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).tolist()
            confidences = probs.max(dim=1).values.tolist()

        # Remap predictions to original labels (0 and 4)
        preds_remapped = [0 if p == 0 else 4 for p in preds]
        return [
            {"tweet": tweet, "prediction": p, "confidence": c}
            for tweet, p, c in zip(texts, preds_remapped, confidences)
        ]

# --- Training function ---
def train_text_classifier_with_mlflow(df,
                                     text_col,
                                     target_col,
                                     mlflow_experiment_name,
                                     embedding_path,
                                     num_epochs=10,
                                     batch_size=32,
                                     lr=0.001,
                                     embedding_dim=300):

    # Load embeddings
    embeddings_index = load_glove_embeddings(embedding_path, embedding_dim)

    # Encode targets: 0 -> 0, 4 -> 1
    df = df.copy()
    df[target_col] = df[target_col].apply(lambda x: 0 if x == 0 else 1)

    # Prepare data
    X = df[text_col].values
    y = df[target_col].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_embed = torch.tensor([embed_text(t, embeddings_index, embedding_dim) for t in X_train], dtype=torch.float32)
    X_test_embed = torch.tensor([embed_text(t, embeddings_index, embedding_dim) for t in X_test], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    num_classes = 2
    model = SimpleClassifier(embedding_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train_embed.size()[0])
        for i in range(0, X_train_embed.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_embed[indices], y_train_tensor[indices]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} done.")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_embed)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()
        print(f"Test Accuracy: {accuracy:.4f}")

    # Exemple d'entrée textuelle pour la signature
    input_example = pd.DataFrame({text_col: ["This is an example tweet."]})

    # Signature
    wrapped_model = TextClassifierWrapper(model, embeddings_index, embedding_dim)
    output_example = wrapped_model.predict(None, input_example)
    signature = infer_signature(input_example, pd.DataFrame(output_example))

    # Logging dans MLflow
    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run():
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_metric("test_accuracy", accuracy)
        print("📦 Logging model with text input...")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,
            input_example=input_example,
            signature=signature,
            registered_model_name="glove_tweet_classifier"
        )

    return model, embeddings_index



import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Supprime les logs INFO, WARNING, et ERROR de TensorFlow

import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW  
from tqdm import tqdm

import mlflow.pyfunc


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class BertWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.model.load_state_dict(torch.load(context.artifacts["model_path"], map_location="cpu"))
        self.model.eval()

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame):
        texts = model_input["text"].tolist()
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)
            confidences = probs.max(dim=1).values.tolist()
            preds = torch.argmax(probs, dim=1).tolist()

        # Remap predictions (si besoin, ici 0 et 1 -> 0 et 4)
        preds_remapped = [0 if p == 0 else 4 for p in preds]

        # Retourner liste de dict avec tweet original, prédiction et confiance
        results = [
            {"tweet": tweet, "prediction": pred, "confidence": conf}
            for tweet, pred, conf in zip(texts, preds_remapped, confidences)
        ]
        return results


def train_bert_classifier_with_mlflow(
    df,
    text_col="text",
    target_col="target",
    mlflow_experiment_name="bert_tweet_classification",
    num_epochs=3,
    batch_size=16,
    lr=2e-5,
    max_len=128
):
    # Nettoyage
    df = df[df[target_col].isin([0, 4])].copy()
    df[target_col] = df[target_col].map({0: 0, 4: 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].tolist(), df[target_col].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = TweetDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = TweetDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "max_length": max_len,
            "text_col": text_col,
            "target_col": target_col,
        })

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in loop:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                true_labels = batch["labels"].numpy()
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                predicted = torch.argmax(output.logits, axis=1).cpu().numpy()
                preds.extend(predicted)
                labels.extend(true_labels)

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Save the model state
        model_path = "bert_model.pt"
        torch.save(model.state_dict(), model_path)

        # Signature
        input_example = pd.DataFrame({text_col: ["I love this tweet!"]})
        with torch.no_grad():
            encoded = tokenizer(input_example[text_col].tolist(), return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**encoded)
            preds_example = torch.argmax(outputs.logits, axis=1).cpu().numpy()

        signature = infer_signature(input_example, preds_example)

        # Log modèle pyfunc
        mlflow.pyfunc.log_model(
            artifact_path="bert_model",
            python_model=BertWrapper(),
            artifacts={"model_path": model_path},
            input_example=input_example,
            signature=signature,
            registered_model_name="bert_tweet_classifier"
        )

        print(f"✅ Fine-tuning terminé — Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        return model, {"accuracy": acc, "f1_score": f1}



import pandas as pd
import matplotlib.pyplot as plt

def tracer_pie_bar(df, colonnes):
    for col in colonnes:
        # Vérifie si la colonne existe dans le DataFrame
        if col not in df.columns:
            print(f"Colonne '{col}' non trouvée dans le DataFrame.")
            continue
        
        # Obtenir la fréquence des valeurs uniques dans la colonne
        compte_valeurs = df[col].value_counts()

        # Tracer le diagramme circulaire
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        compte_valeurs.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title(f"Diagramme circulaire pour la variable {col}")
        plt.ylabel('')

        # Tracer le diagramme en barres
        plt.subplot(1, 2, 2)
        compte_valeurs.plot.bar(color='skyblue')
        plt.title(f"Diagramme en barres pour la variable {col}")
        plt.xlabel(col)
        plt.ylabel("Nombre d'individus")

        # Afficher les deux graphiques
        plt.tight_layout()
        plt.show()



import os
import re
import json
import mlflow
import mlflow.pyfunc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from mlflow.models.signature import infer_signature
from gensim.models import KeyedVectors
from mlflow.tracking.client import MlflowClient
from mlflow.exceptions import MlflowException

# --- Model Definition ---
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- Load Reduced Word2Vec embeddings ---
def load_reduced_word2vec(embedding_path, embedding_dim=None):
    """Charge un modèle Word2Vec réduit avec gestion des formats .model et .npy"""
    if embedding_path.endswith('.model'):
        # Format Gensim KeyedVectors
        model = KeyedVectors.load(embedding_path)
        embeddings_index = {word: model[word] for word in model.key_to_index}
        actual_dim = model.vector_size
    elif embedding_path.endswith('.npy'):
        # Format numpy array + fichier vocab séparé
        vectors = np.load(embedding_path)
        with open(embedding_path.replace('.npy', '.vocab'), 'r') as f:
            vocab = [line.strip() for line in f]
        embeddings_index = {word: vectors[i] for i, word in enumerate(vocab)}
        actual_dim = vectors.shape[1]
    else:
        raise ValueError("Format non supporté. Utilisez .model ou .npy")
    
    if embedding_dim is not None and embedding_dim != actual_dim:
        print(f"Attention: La dimension demandée ({embedding_dim}) ne correspond pas à celle du modèle ({actual_dim})")
    
    return embeddings_index, actual_dim

# --- Optimized Text Embedding ---
def batch_embed(texts, embeddings_index, embedding_dim, batch_size=1000):
    """Calcule les embeddings par batch pour économiser la mémoire"""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors = []
        for text in batch:
            tokens = simple_tokenize(text)
            word_vectors = [embeddings_index.get(w, np.zeros(embedding_dim)) for w in tokens]
            text_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(embedding_dim)
            vectors.append(text_vector)
        all_vectors.extend(vectors)
    return np.array(all_vectors)

# --- Simple tokenizer ---
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# --- Optimized Wrapper PyFunc ---
class TextClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, embeddings_index, embedding_dim=100):
        self.model = model
        self.embeddings_index = embeddings_index
        self.embedding_dim = embedding_dim

    def _preprocess(self, text_series):
        vectors = batch_embed(text_series.tolist(), self.embeddings_index, self.embedding_dim)
        return torch.tensor(vectors, dtype=torch.float32)

    def predict(self, context, model_input):
        texts = model_input["text"]
        X_embed = self._preprocess(texts)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_embed)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).tolist()
            confidences = probs.max(dim=1).values.tolist()

        # Remap predictions to original labels (0 and 4)
        preds_remapped = [0 if p == 0 else 4 for p in preds]
        return [
            {"tweet": tweet, "prediction": p, "confidence": c}
            for tweet, p, c in zip(texts, preds_remapped, confidences)
        ]

# --- Training function for reduced model ---
def train_with_reduced_word2vec(df,
                              text_col,
                              target_col,
                              mlflow_experiment_name,
                              embedding_path,
                              num_epochs=10,
                              batch_size=32,
                              lr=0.001):
    
    # Initialisation MLflow avec vérification
    mlflow.set_tracking_uri("http://127.0.0.1:8080")  # Ajustez selon votre configuration
    client = MlflowClient()
    
    try:
        # Création de l'expérience si elle n'existe pas
        if not mlflow.get_experiment_by_name(mlflow_experiment_name):
            mlflow.create_experiment(mlflow_experiment_name)
            print(f"Création de l'expérience {mlflow_experiment_name}")
    except MlflowException as e:
        print(f"Erreur lors de la création de l'expérience: {str(e)}")
        raise

    # Load reduced embeddings
    embeddings_index, embedding_dim = load_reduced_word2vec(embedding_path)
    
    # Prepare data
    df = df.copy()
    df[target_col] = df[target_col].apply(lambda x: 0 if x == 0 else 1)  # Encode targets
    
    X = df[text_col].values
    y = df[target_col].values
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Embedding with memory optimization
    print("Embedding training texts...")
    X_train_embed = torch.tensor(
        batch_embed(X_train, embeddings_index, embedding_dim), 
        dtype=torch.float32
    )
    
    print("Embedding test texts...")
    X_test_embed = torch.tensor(
        batch_embed(X_test, embeddings_index, embedding_dim), 
        dtype=torch.float32
    )
    
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Model training
    model = SimpleClassifier(embedding_dim, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train_embed.size()[0])
        for i in range(0, X_train_embed.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_embed[indices], y_train_tensor[indices]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_embed)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()
        print(f"Test Accuracy: {accuracy:.4f}")
    
    # MLflow logging avec gestion robuste des erreurs
    mlflow.set_experiment(mlflow_experiment_name)
    
    try:
        with mlflow.start_run() as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            
            # Préparation de l'exemple et signature
            input_example = pd.DataFrame({text_col: ["This is an example tweet for signature inference"]})
            wrapped_model = TextClassifierWrapper(model, embeddings_index, embedding_dim)
            output_example = wrapped_model.predict(None, input_example)
            signature = infer_signature(input_example, pd.DataFrame(output_example))
            
            # Log des paramètres et métriques
            mlflow.log_params({
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "embedding_type": "reduced_word2vec",
                "embedding_dim": embedding_dim,
                "embedding_source": os.path.basename(embedding_path)
            })
            
            mlflow.log_metrics({
                "test_accuracy": accuracy,
                "final_loss": loss.item()
            })
            
            # Enregistrement du modèle avec timeout augmenté
            print("Début de l'enregistrement du modèle dans MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapped_model,
                input_example=input_example,
                signature=signature,
                registered_model_name="reduced_w2v_classifier",
                await_registration_for=300  # Attendre 5 minutes max
            )
            
            # Vérification de l'enregistrement
            try:
                latest_version = client.get_latest_versions("reduced_w2v_classifier", stages=["None"])[0]
                print(f"Modèle enregistré avec succès. Version {latest_version.version}")
                mlflow.log_text(f"Modèle enregistré: version {latest_version.version}", "registration_status.txt")
            except Exception as e:
                print(f"Attention: Impossible de vérifier l'enregistrement du modèle: {str(e)}")
            
            # Sauvegarde d'artefacts supplémentaires
            mlflow.log_dict({"accuracy": accuracy, "loss": loss.item()}, "metrics.json")
            print(f"Run {run.info.run_id} complété avec succès")
            
    except Exception as e:
        print(f"Erreur critique lors de l'enregistrement MLflow: {str(e)}")
        if 'run' in locals():
            mlflow.end_run(status="FAILED")
        raise
    
    return model, embeddings_index