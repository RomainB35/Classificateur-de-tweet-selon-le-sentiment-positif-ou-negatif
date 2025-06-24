import pandas as pd
import plotly.express as px
import plotly.io as pio

# Configure l'affichage Plotly dans Jupyter
pio.renderers.default = 'notebook'  # ou 'iframe_connected'

def plot_sampled_embeddings(df, target_col, embedding_cols, sample_size=1000, random_state=42):
    """
    Affiche une projection 2D ou 3D à partir de colonnes d'embedding avec coloration par target.

    Paramètres :
    - df : DataFrame contenant les données
    - target_col : nom de la colonne cible (ex. 'target')
    - embedding_cols : liste des noms des colonnes d'embedding (2 ou 3 colonnes : ['x', 'y'] ou ['x', 'y', 'z'])
    - sample_size : taille de l’échantillon à visualiser
    - random_state : pour reproductibilité
    """

    # Vérification des colonnes nécessaires
    missing_cols = [col for col in embedding_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    # Échantillonnage
    df_sampled = df.sample(n=min(sample_size, len(df)), random_state=random_state).copy()

    # Couleur qualitative ou continue ?
    color_scale = 'Viridis' if df_sampled[target_col].nunique() > 5 else px.colors.qualitative.Set1

    # Affichage 2D ou 3D selon le nombre de colonnes
    if len(embedding_cols) == 2:
        fig = px.scatter(
            df_sampled,
            x=embedding_cols[0],
            y=embedding_cols[1],
            color=target_col,
            hover_data=['id', 'text'] if 'id' in df.columns and 'text' in df.columns else None,
            title="Projection 2D des embeddings (échantillon)",
            color_continuous_scale=color_scale if isinstance(color_scale, str) else None,
            color_discrete_sequence=color_scale if isinstance(color_scale, list) else None
        )
    elif len(embedding_cols) == 3:
        fig = px.scatter_3d(
            df_sampled,
            x=embedding_cols[0],
            y=embedding_cols[1],
            z=embedding_cols[2],
            color=target_col,
            hover_data=['id', 'text'] if 'id' in df.columns and 'text' in df.columns else None,
            title="Projection 3D des embeddings (échantillon)",
            color_continuous_scale=color_scale if isinstance(color_scale, str) else None,
            color_discrete_sequence=color_scale if isinstance(color_scale, list) else None
        )
    else:
        raise ValueError("embedding_cols doit contenir 2 ou 3 colonnes.")

    fig.show()
