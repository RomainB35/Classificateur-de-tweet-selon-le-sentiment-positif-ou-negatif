import os
import sys
import ctypes
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(old_stderr_fd)
        os.dup2(devnull.fileno(), old_stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, old_stderr_fd)

with suppress_stderr():
    import tensorflow as tf


import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Téléchargement des ressources NLTK (à décommenter si nécessaire)
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text_columns(df, text_column):
    """
    Crée un nouveau DataFrame avec 4 colonnes supplémentaires de texte prétraité
    
    Args:
        df: DataFrame pandas original (non modifié)
        text_column: Nom de la colonne contenant le texte à traiter
    
    Returns:
        Nouveau DataFrame avec les colonnes originales + 4 nouvelles colonnes:
        - '[text_column]_bow' : Pour Bag-of-Words/TF-IDF
        - '[text_column]_w2v' : Pour Word2Vec/GloVe/FastText
        - '[text_column]_bert' : Pour BERT
        - '[text_column]_use' : Pour Universal Sentence Encoder (USE)
    """
    
    # Création d'une copie indépendante du DataFrame
    new_df = df.copy()
    
    # Initialisation des outils
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'of', 'the', 'in', 'is', 'it', 'and', 'this', 'for', 'to', 'with'}
    
    # Fonction pour BoW/TF-IDF
    def _process_bow(text):
        if pd.isna(text):
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    # Fonction pour Word2Vec
    def _process_w2v(text):
        if pd.isna(text):
            return ''
        text = str(text)
        text = re.sub(r'(\d+\s?cm)', ' DIM_CM ', text)
        text = re.sub(r'Rs\.\s?\d+', ' PRICE_RS ', text)
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        tokens = [word.lower() for word in tokens if word.lower() not in custom_stopwords and len(word) > 1]
        
        # Fusion des termes composés
        merged_tokens = []
        skip = False
        for i in range(len(tokens)):
            if skip:
                skip = False
                continue
            if i < len(tokens)-1 and tokens[i] == 'eyelet' and tokens[i+1] in ['door', 'curtain']:
                merged_tokens.append(f'{tokens[i]}_{tokens[i+1]}')
                skip = True
            else:
                merged_tokens.append(tokens[i])
        
        return ' '.join(merged_tokens)
    
    # Fonction pour BERT
    def _process_bert(text):
        if pd.isna(text):
            return ''
        text = str(text)
        text = re.sub(r'(\d+\s?cm)', '[DIM]', text)
        text = re.sub(r'Rs\.\s?\d+', '[PRICE]', text)
        text = re.sub(r'Model ID \w+', 'Model ID [ID]', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Fonction pour USE (Universal Sentence Encoder)
    def _process_use(text):
        if pd.isna(text):
            return ''
        text = str(text)
        # Nettoyage de base pour USE
        text = re.sub(r'\s+', ' ', text).strip()
        # Conservation des informations importantes
        text = re.sub(r'(\d+\s?cm)', '\\1 ', text)  # Garde les dimensions
        text = re.sub(r'(Rs\.\s?\d+)', '\\1 ', text)  # Garde les prix
        # Suppression des caractères spéciaux inutiles
        text = re.sub(r'[^\w\s.,;:!?\'"-]', ' ', text)
        return text
    
    # Ajout des nouvelles colonnes
    new_df[f'{text_column}_bow'] = df[text_column].apply(_process_bow)
    new_df[f'{text_column}_w2v'] = df[text_column].apply(_process_w2v)
    new_df[f'{text_column}_bert'] = df[text_column].apply(_process_bert)
    new_df[f'{text_column}_use'] = df[text_column].apply(_process_use)
    
    return new_df

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import vstack, hstack, csr_matrix
import numpy as np

def extract_bow_tfidf_features_from_dataframe(
    df, 
    text_col,
    add_2d_projection=True,
    add_3d_projection=True,
    max_features=5000,  # Réduction drastique par défaut
    ngram_range=(1, 1),
    batch_size=10000,
    pca_components=300,  # Réduction avant projection finale
    include_raw_features=False  # Nouveau paramètre pour inclure les features brutes
):
    """
    Version optimisée pour gérer les gros datasets sans problèmes de mémoire.
    Nouveauté : possibilité d'inclure les features brutes (BoW et TF-IDF)
    """
    # Création d'une copie du DataFrame original
    new_df = df.copy()
    
    # S'assurer qu'aucune valeur manquante n'est présente
    texts = new_df[text_col].fillna('').astype(str).tolist()
    
    # 1. Initialisation des vectorizers avec paramètres optimisés
    bow_vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        dtype=np.float32  # Réduction de la taille mémoire
    )
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        dtype=np.float32
    )
    
    # 2. Fit des vectorizers
    print("Fitting vectorizers...")
    bow_vectorizer.fit(texts)
    tfidf_vectorizer.fit(texts)
    
    # 3. Fonction pour traiter par batch avec réduction de dimension
    def process_and_reduce(vectorizer, texts, batch_size, return_raw=False):
        print("Processing batches with dimensionality reduction...")
        svd = TruncatedSVD(n_components=pca_components)
        
        # Initialisation des variables
        reduced_features_list = []
        raw_features_list = [] if return_raw else None
        
        # Traitement par lots
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_features = vectorizer.transform(batch)
            
            if return_raw:
                raw_features_list.append(batch_features)
            
            if i == 0:
                svd.fit(batch_features)
                reduced_batch = svd.transform(batch_features)
            else:
                reduced_batch = svd.transform(batch_features)
            
            reduced_features_list.append(reduced_batch)
        
        # Concaténation des résultats
        reduced_features = np.vstack(reduced_features_list)
        
        if return_raw:
            raw_features = vstack(raw_features_list) if raw_features_list else None
            return reduced_features, svd, raw_features
        else:
            return reduced_features, svd, None
    
    # 4. Traitement BoW et TF-IDF
    print("Processing BoW features...")
    if include_raw_features:
        bow_reduced, bow_svd, bow_raw = process_and_reduce(bow_vectorizer, texts, batch_size, True)
    else:
        bow_reduced, bow_svd, _ = process_and_reduce(bow_vectorizer, texts, batch_size)
    
    print("Processing TF-IDF features...")
    if include_raw_features:
        tfidf_reduced, tfidf_svd, tfidf_raw = process_and_reduce(tfidf_vectorizer, texts, batch_size, True)
    else:
        tfidf_reduced, tfidf_svd, _ = process_and_reduce(tfidf_vectorizer, texts, batch_size)
    
    # 5. Ajout des projections finales (2D/3D)
    def add_projections(features, prefix, df):
        # Projection 2D
        if add_2d_projection:
            pca_2d = PCA(n_components=2)
            projections_2d = pca_2d.fit_transform(features)
            df[f'{prefix}_2d_x'] = projections_2d[:, 0]
            df[f'{prefix}_2d_y'] = projections_2d[:, 1]
        
        # Projection 3D
        if add_3d_projection:
            pca_3d = PCA(n_components=3)
            projections_3d = pca_3d.fit_transform(features)
            df[f'{prefix}_3d_x'] = projections_3d[:, 0]
            df[f'{prefix}_3d_y'] = projections_3d[:, 1]
            df[f'{prefix}_3d_z'] = projections_3d[:, 2]
        
        return df
    
    print("Adding projections...")
    new_df = add_projections(bow_reduced, 'bow', new_df)
    new_df = add_projections(tfidf_reduced, 'tfidf', new_df)
    
    # 6. Ajout des features brutes si demandé
    if include_raw_features:
        print("Adding raw features...")
        # Conversion des matrices creuses en DataFrames
        bow_df = pd.DataFrame.sparse.from_spmatrix(bow_raw, columns=[f'bow_raw_{i}' for i in range(bow_raw.shape[1])])
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_raw, columns=[f'tfidf_raw_{i}' for i in range(tfidf_raw.shape[1])])
        
        # Concaténation avec le DataFrame principal
        new_df = pd.concat([new_df, bow_df, tfidf_df], axis=1)
    
    print("Processing complete!")
    
    # Retour des résultats
    if include_raw_features:
        return new_df, bow_vectorizer, tfidf_vectorizer, bow_svd, tfidf_svd, bow_raw, tfidf_raw
    else:
        return new_df, bow_vectorizer, tfidf_vectorizer, bow_svd, tfidf_svd


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def tracer_densite_boxplot_df(df, variables):
    for var in variables:
        if var in df.columns:
            # Extraire la série
            series = df[var].dropna()  # Supprime les valeurs manquantes

            # Créer une figure avec deux sous-graphes
            plt.figure(figsize=(12, 5))

            # Courbe de densité (KDE)
            plt.subplot(1, 2, 1)
            sns.kdeplot(series, fill=True, color="blue")
            plt.title(f"Courbe de densité pour la variable {var}")

            # Boîte à moustaches (boxplot)
            plt.subplot(1, 2, 2)
            sns.boxplot(data=series, orient='h', color="skyblue")
            plt.title(f"Boîte à moustache pour la variable {var}")

            # Ajuster l'espacement pour éviter le chevauchement
            plt.tight_layout()
            plt.show()

            compute_statistics_variable(df, var)
        else:
            print(f"La variable '{var}' n'existe pas dans le dataframe.")     


import pandas as pd
import matplotlib.pyplot as plt

def tracer_pie_bar_top10_autre(df, colonnes):
    for col in colonnes:
        # Vérifie si la colonne existe dans le DataFrame
        if col not in df.columns:
            print(f"Colonne '{col}' non trouvée dans le DataFrame.")
            continue

        # Obtenir les 10 premières valeurs les plus fréquentes
        top10_values = df[col].value_counts().nlargest(10)

        # Calculer le reste des valeurs et les regrouper sous 'Autre'
        autres_sum = df[col].value_counts().iloc[10:].sum()
        if autres_sum > 0:
            top10_values['Autre'] = autres_sum

        # Tracer le diagramme circulaire
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        top10_values.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title(f"Top 10 + Autre - Diagramme circulaire de {col}")
        plt.ylabel('')

        # Tracer l'histogramme (bar chart)
        plt.subplot(1, 2, 2)
        top10_values.plot.bar(color='skyblue')
        plt.title(f"Top 10 + Autre - Histogramme de {col}")
        plt.xlabel(col)
        plt.ylabel('Fréquence')

        # Ajuster l'espacement pour éviter le chevauchement
        plt.tight_layout()
        plt.show()  


import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import warnings
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
warnings.filterwarnings("ignore")

def extract_classic_embeddings(
    df, 
    text_col, 
    embedding_type='word2vec', 
    embedding_path=None, 
    dim=300, 
    batch_size=1000,
    add_2d_projection=True,
    add_3d_projection=True,
    normalize_embeddings=True
):
    """
    Fonction optimisée pour extraire des embeddings (Word2Vec, GloVe ou FastText) avec projections PCA.
    Version modifiée avec traitement par lots optimisé.
    
    Paramètres:
        df: DataFrame pandas
        text_col: Nom de la colonne texte
        embedding_type: 'word2vec', 'glove' ou 'fasttext'
        embedding_path: Chemin vers le modèle pré-entraîné
        dim: Dimension des embeddings
        batch_size: Nombre de textes à traiter simultanément
        add_2d_projection: Si True, ajoute une projection PCA 2D
        add_3d_projection: Si True, ajoute une projection PCA 3D
        normalize_embeddings: Si True, normalise les embeddings avant calcul
        
    Retourne:
        DataFrame enrichi avec les embeddings et leurs projections
    """
    # ========== VALIDATION DES ENTREES ==========
    if embedding_type not in ['word2vec', 'glove', 'fasttext']:
        raise ValueError("Type d'embedding non supporté. Choisir parmi: 'word2vec', 'glove', 'fasttext'")
    
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("La dimension doit être un entier positif")
        
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Fichier '{embedding_path}' introuvable")
    
    # ========== CHARGEMENT DES MODELES ==========
    print("Chargement du modèle d'embedding...")
    try:
        if embedding_type == 'word2vec':
            try:
                model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
            except:
                model = KeyedVectors.load(embedding_path)
                
        elif embedding_type == 'glove':
            tmp_file = "glove_w2v_format.txt"
            glove2word2vec(embedding_path, tmp_file)
            model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
            
        elif embedding_type == 'fasttext':
            if embedding_path.endswith('.bin'):
                raise ValueError("Utilisez le format .vec pour FastText. Pour .bin, utilisez la librairie fasttext directement")
            elif embedding_path.endswith('.vec'):
                model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
            else:
                raise ValueError("Format FastText non reconnu. Utilisez .vec ou .bin")
    
    except Exception as e:
        raise ValueError(f"Erreur de chargement du modèle {embedding_type}: {str(e)}")
    
    # ========== EXTRACTION DES EMBEDDINGS ==========
    result_df = df.copy()
    texts = result_df[text_col].astype(str).tolist()
    
    # Initialisation du tableau numpy pour stocker les embeddings
    embeddings_array = np.zeros((len(texts), dim), dtype=np.float32)
    
    # Fonction pour traiter un batch de textes
    def process_batch(batch_texts, start_idx):
        batch_embeddings = np.zeros((len(batch_texts), dim), dtype=np.float32)
        
        for i, text in enumerate(batch_texts):
            words = text.lower().split()
            valid_words = [word for word in words if word in model.key_to_index]
            
            if valid_words:
                embedding = np.mean([model[word] for word in valid_words], axis=0)
                if normalize_embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                batch_embeddings[i] = embedding
        
        return batch_embeddings
    
    # Traitement par batch avec barre de progression
    print("Extraction des embeddings par lots...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Traitement des textes"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = process_batch(batch_texts, i)
        embeddings_array[i:i+batch_size] = batch_embeddings
    
    # Stockage des embeddings dans le DataFrame
    result_df[f'embedding_{embedding_type}'] = list(embeddings_array)
    
    # ========== PROJECTIONS DIMENSIONNELLES ==========
    def add_pca_projections(features_array, prefix, df):
        # Ne garder que les embeddings non nuls
        non_zero_mask = np.any(features_array != 0, axis=1)
        valid_features = features_array[non_zero_mask]
        
        if len(valid_features) > 1:  # PCA nécessite au moins 2 échantillons
            try:
                # Projection 2D
                if add_2d_projection:
                    print("Calcul des projections PCA 2D...")
                    pca_2d = PCA(n_components=2)
                    projections_2d = pca_2d.fit_transform(valid_features)
                    
                    # Initialisation des colonnes avec des valeurs par défaut
                    df[f'{prefix}_2d_x'] = np.nan
                    df[f'{prefix}_2d_y'] = np.nan
                    
                    # Remplissage pour les indices valides
                    valid_indices = np.where(non_zero_mask)[0]
                    df.loc[valid_indices, f'{prefix}_2d_x'] = projections_2d[:, 0]
                    df.loc[valid_indices, f'{prefix}_2d_y'] = projections_2d[:, 1]
                
                # Projection 3D
                if add_3d_projection:
                    print("Calcul des projections PCA 3D...")
                    pca_3d = PCA(n_components=3)
                    projections_3d = pca_3d.fit_transform(valid_features)
                    
                    df[f'{prefix}_3d_x'] = np.nan
                    df[f'{prefix}_3d_y'] = np.nan
                    df[f'{prefix}_3d_z'] = np.nan
                    
                    valid_indices = np.where(non_zero_mask)[0]
                    df.loc[valid_indices, f'{prefix}_3d_x'] = projections_3d[:, 0]
                    df.loc[valid_indices, f'{prefix}_3d_y'] = projections_3d[:, 1]
                    df.loc[valid_indices, f'{prefix}_3d_z'] = projections_3d[:, 2]
                        
            except Exception as e:
                print(f"Erreur lors de la PCA: {str(e)}")
        return df
    
    # Ajout des projections PCA
    print("Ajout des projections dimensionnelles...")
    result_df = add_pca_projections(embeddings_array, f'embedding_{embedding_type}', result_df)
    
    print("Traitement terminé avec succès!")
    return result_df

import torch
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import Optional

def add_bert_features(
    df: pd.DataFrame,
    text_column: str,
    output_column: str = 'features_BERT',
    model_name: str = 'bert-base-uncased',
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[str] = None,
    disable_progress: bool = False,
    add_2d_projection: bool = True,
    add_3d_projection: bool = False,
    normalize_embeddings: bool = True,
    layer_pooling: str = 'mean'  # 'mean', 'cls', or 'max'
) -> pd.DataFrame:
    """
    Ajoute des embeddings BERT à un DataFrame avec options de projections dimensionnelles.
    
    Args:
        df: DataFrame pandas contenant les données textuelles
        text_column: Nom de la colonne contenant le texte
        output_column: Nom de la colonne de sortie pour les features
        model_name: Nom du modèle BERT (par défaut 'bert-base-uncased')
        batch_size: Taille des lots pour le traitement
        max_length: Longueur maximale des séquences
        device: Device pour le calcul ('cuda' ou 'cpu'). Si None, détecte automatiquement.
        disable_progress: Désactive la barre de progression
        add_2d_projection: Si True, ajoute une projection PCA 2D
        add_3d_projection: Si True, ajoute une projection PCA 3D
        normalize_embeddings: Si True, normalise les embeddings
        layer_pooling: Méthode de pooling ('mean', 'cls', ou 'max')
    
    Returns:
        Un nouveau DataFrame avec les embeddings et leurs projections
    """
    # Vérification des entrées
    if text_column not in df.columns:
        raise ValueError(f"La colonne '{text_column}' n'existe pas dans le DataFrame")
    
    if layer_pooling not in ['mean', 'cls', 'max']:
        raise ValueError("layer_pooling doit être 'mean', 'cls' ou 'max'")
    
    # Configuration du device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Chargement du modèle et tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Préparation des données
    texts = df[text_column].astype(str).tolist()
    num_texts = len(texts)
    embeddings = []
    
    # Traitement par batch avec barre de progression
    for i in tqdm(range(0, num_texts, batch_size), 
                  disable=disable_progress, 
                  desc="Extraction BERT"):
        batch = texts[i:i + batch_size]
        
        # Tokenization
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        ).to(device)
        
        # Calcul des embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Pooling des embeddings
        last_hidden = outputs.last_hidden_state
        
        if layer_pooling == 'mean':
            batch_embeddings = last_hidden.mean(dim=1)
        elif layer_pooling == 'cls':
            batch_embeddings = last_hidden[:, 0, :]  # Token [CLS]
        elif layer_pooling == 'max':
            batch_embeddings = last_hidden.max(dim=1).values
        
        if normalize_embeddings:
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concaténation des résultats
    all_embeddings = np.vstack(embeddings)
    
    # Création d'un nouveau DataFrame
    result_df = df.copy()
    result_df[output_column] = list(all_embeddings)
    
    # Ajout des projections dimensionnelles
    def add_pca_projections(embeddings_array, prefix, df):
        # Ne garder que les embeddings non nuls
        non_zero_mask = np.array([np.any(e != 0) for e in embeddings_array])
        valid_embeddings = embeddings_array[non_zero_mask]
        
        if len(valid_embeddings) > 1:  # PCA nécessite au moins 2 échantillons
            try:
                # Projection 2D
                if add_2d_projection:
                    pca_2d = PCA(n_components=2)
                    projections_2d = pca_2d.fit_transform(valid_embeddings)
                    
                    df[f'{prefix}_2d_x'] = None
                    df[f'{prefix}_2d_y'] = None
                    
                    valid_indices = np.where(non_zero_mask)[0]
                    for i, idx in enumerate(valid_indices):
                        df.at[idx, f'{prefix}_2d_x'] = projections_2d[i, 0]
                        df.at[idx, f'{prefix}_2d_y'] = projections_2d[i, 1]
                
                # Projection 3D
                if add_3d_projection:
                    pca_3d = PCA(n_components=3)
                    projections_3d = pca_3d.fit_transform(valid_embeddings)
                    
                    df[f'{prefix}_3d_x'] = None
                    df[f'{prefix}_3d_y'] = None
                    df[f'{prefix}_3d_z'] = None
                    
                    valid_indices = np.where(non_zero_mask)[0]
                    for i, idx in enumerate(valid_indices):
                        df.at[idx, f'{prefix}_3d_x'] = projections_3d[i, 0]
                        df.at[idx, f'{prefix}_3d_y'] = projections_3d[i, 1]
                        df.at[idx, f'{prefix}_3d_z'] = projections_3d[i, 2]
                        
            except Exception as e:
                print(f"Erreur lors de la PCA: {str(e)}")
        return df
    
    result_df = add_pca_projections(all_embeddings, output_column, result_df)
    
    return result_df


import os
import sys
import ctypes
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(old_stderr_fd)
        os.dup2(devnull.fileno(), old_stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, old_stderr_fd)

with suppress_stderr():
    import tensorflow as tf


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import logging
from typing import Union, Optional
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configure environment for silent CPU-only operation
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

def add_use_embeddings(
    df: pd.DataFrame,
    text_column: str,
    model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4",
    batch_size: int = 32,
    use_lite: bool = False,
    show_warnings: bool = False,
    add_2d_projection: bool = True,
    add_3d_projection: bool = False,
    normalize_embeddings: bool = True,
    output_column: Optional[str] = None,
    disable_progress: bool = False,
    pca_variance_threshold: Optional[float] = None  # <<< Nouveau paramètre
) -> pd.DataFrame:
    """
    Adds USE embeddings to DataFrame with optimized batch processing and memory management.

    Args:
        df: Input DataFrame
        text_column: Column containing text to embed
        model_url: TF Hub model URL
        batch_size: Processing batch size (default: 32)
        use_lite: Use lite model version (default: False)
        show_warnings: Show TensorFlow warnings (default: False)
        add_2d_projection: Add 2D PCA projection (default: True)
        add_3d_projection: Add 3D PCA projection (default: False)
        normalize_embeddings: Normalize embeddings (default: True)
        output_column: Output column name (default: '[text_column]_use_embedding')
        disable_progress: Disable progress bar (default: False)
        pca_variance_threshold: If set, apply PCA to retain given variance (e.g. 0.95)

    Returns:
        DataFrame with embeddings and optional projections
    """
    if not show_warnings:
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)

    tf.config.set_visible_devices([], 'GPU')

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")

    output_column = output_column or f"{text_column}_use_embedding"

    if use_lite:
        model_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"

    print("Loading USE model...")
    try:
        with tf.device('/CPU:0'):
            model = hub.load(model_url)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

    new_df = df.copy()
    texts = new_df[text_column].astype(str).tolist()

    sample_embedding = model([""]).numpy()[0]
    embedding_dim = len(sample_embedding)
    embeddings_array = np.zeros((len(texts), embedding_dim), dtype=np.float32)

    print("Processing text embeddings...")
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="Generating embeddings",
                  disable=disable_progress,
                  unit="batch"):
        batch = texts[i:i + batch_size]
        try:
            with tf.device('/CPU:0'):
                batch_embeddings = model(batch).numpy()

                if normalize_embeddings:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    batch_embeddings = batch_embeddings / norms

                embeddings_array[i:i + batch_size] = batch_embeddings
        except Exception as e:
            raise RuntimeError(f"Batch processing failed at index {i}: {str(e)}")

    new_df[output_column] = list(embeddings_array)

    def add_projections(features, prefix, df):
        non_zero_mask = np.any(features != 0, axis=1)
        valid_features = features[non_zero_mask]

        if len(valid_features) > 1:
            try:
                valid_idx = np.where(non_zero_mask)[0]

                if pca_variance_threshold is not None:
                    pca = PCA(n_components=pca_variance_threshold)
                    proj = pca.fit_transform(valid_features)
                    for i in range(proj.shape[1]):
                        col_name = f'{prefix}_pca_{i+1}'
                        df[col_name] = np.nan
                        df.loc[valid_idx, col_name] = proj[:, i]
                else:
                    if add_2d_projection:
                        pca_2d = PCA(n_components=2)
                        proj_2d = pca_2d.fit_transform(valid_features)
                        df[f'{prefix}_2d_x'] = np.nan
                        df[f'{prefix}_2d_y'] = np.nan
                        df.loc[valid_idx, f'{prefix}_2d_x'] = proj_2d[:, 0]
                        df.loc[valid_idx, f'{prefix}_2d_y'] = proj_2d[:, 1]

                    if add_3d_projection:
                        pca_3d = PCA(n_components=3)
                        proj_3d = pca_3d.fit_transform(valid_features)
                        df[f'{prefix}_3d_x'] = np.nan
                        df[f'{prefix}_3d_y'] = np.nan
                        df[f'{prefix}_3d_z'] = np.nan
                        df.loc[valid_idx, f'{prefix}_3d_x'] = proj_3d[:, 0]
                        df.loc[valid_idx, f'{prefix}_3d_y'] = proj_3d[:, 1]
                        df.loc[valid_idx, f'{prefix}_3d_z'] = proj_3d[:, 2]
            except Exception as e:
                if show_warnings:
                    print(f"PCA warning: {str(e)}")

        return df

    print("Adding dimensional projections...")
    new_df = add_projections(embeddings_array, output_column, new_df)

    print("Embedding generation complete!")
    return new_df


import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Needed for BERT preprocessing
from tqdm import tqdm
from sklearn.decomposition import PCA
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(old_stderr_fd)
        os.dup2(devnull.fileno(), old_stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, old_stderr_fd)

def add_bert_embeddings(
    df: pd.DataFrame,
    text_column: str,
    model_url: str = "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",
    preprocess_url: str = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    batch_size: int = 32,
    show_warnings: bool = False,
    add_2d_projection: bool = True,
    add_3d_projection: bool = False,
    normalize_embeddings: bool = True,
    output_column: str = None,
    disable_progress: bool = False,
    pca_variance_threshold: float = None,
    sample_size: int = None
) -> pd.DataFrame:
    """Adds BERT embeddings and projections to a DataFrame."""

    if not show_warnings:
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)

    tf.config.set_visible_devices([], 'GPU')

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")

    output_column = output_column or f"{text_column}_bert_embedding"

    # Sample if requested
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print("Loading BERT model and preprocessor...")
    try:
        with suppress_stderr():
            preprocess_model = hub.load(preprocess_url)
            encoder_model = hub.load(model_url)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

    texts = df[text_column].astype(str).tolist()

    # Tokenization & preprocessing
    def preprocess_texts(texts_batch):
        return preprocess_model(tf.constant(texts_batch))

    # Encode texts in batches
    sample_output = encoder_model(preprocess_texts(["test"]))
    embedding_dim = sample_output["pooled_output"].shape[-1]
    embeddings_array = np.zeros((len(texts), embedding_dim), dtype=np.float32)

    print("Encoding text with BERT...")
    for i in tqdm(range(0, len(texts), batch_size), disable=disable_progress, desc="BERT encoding"):
        batch = texts[i:i+batch_size]
        try:
            bert_inputs = preprocess_texts(batch)
            outputs = encoder_model(bert_inputs)
            embeddings = outputs["pooled_output"].numpy()

            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms

            embeddings_array[i:i+batch_size] = embeddings
        except Exception as e:
            raise RuntimeError(f"Batch failed at index {i}: {str(e)}")

    df[output_column] = list(embeddings_array)

    def add_projections(features, prefix, df):
        non_zero_mask = np.any(features != 0, axis=1)
        valid_features = features[non_zero_mask]

        if len(valid_features) > 1:
            try:
                valid_idx = np.where(non_zero_mask)[0]
                if pca_variance_threshold is not None:
                    pca = PCA(n_components=pca_variance_threshold)
                    proj = pca.fit_transform(valid_features)
                    for i in range(proj.shape[1]):
                        col = f'{prefix}_pca_{i+1}'
                        df[col] = np.nan
                        df.loc[valid_idx, col] = proj[:, i]
                else:
                    if add_2d_projection:
                        pca2 = PCA(n_components=2)
                        proj2 = pca2.fit_transform(valid_features)
                        df[f'{prefix}_2d_x'] = np.nan
                        df[f'{prefix}_2d_y'] = np.nan
                        df.loc[valid_idx, f'{prefix}_2d_x'] = proj2[:, 0]
                        df.loc[valid_idx, f'{prefix}_2d_y'] = proj2[:, 1]

                    if add_3d_projection:
                        pca3 = PCA(n_components=3)
                        proj3 = pca3.fit_transform(valid_features)
                        df[f'{prefix}_3d_x'] = np.nan
                        df[f'{prefix}_3d_y'] = np.nan
                        df[f'{prefix}_3d_z'] = np.nan
                        df.loc[valid_idx, f'{prefix}_3d_x'] = proj3[:, 0]
                        df.loc[valid_idx, f'{prefix}_3d_y'] = proj3[:, 1]
                        df.loc[valid_idx, f'{prefix}_3d_z'] = proj3[:, 2]
            except Exception as e:
                if show_warnings:
                    print(f"PCA warning: {e}")
        return df

    print("Adding projections...")
    df = add_projections(embeddings_array, output_column, df)

    print("✅ Embedding generation complete.")
    return df
