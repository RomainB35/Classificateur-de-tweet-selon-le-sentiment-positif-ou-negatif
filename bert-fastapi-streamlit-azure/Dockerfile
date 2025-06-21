FROM python:3.10-slim

# Installer dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier fichiers
COPY dockerfiles/bert-fastapi-cpu-streamlit-azure/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Installer torch CPU uniquement
RUN pip install --no-cache-dir torch==2.1.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Installer les autres dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY dockerfiles/bert-fastapi-cpu-streamlit-azure/app.py .
COPY dockerfiles/bert-fastapi-cpu-streamlit-azure/streamlit_app.py .
COPY dockerfiles/bert-fastapi-cpu-streamlit-azure/start.sh .
COPY saved_model/ ./saved_model/

# Rendre start.sh exécutable
RUN chmod +x start.sh

# Exposer les ports (FastAPI + Streamlit)
EXPOSE 8000
EXPOSE 8501

# Commande de démarrage
CMD ["./start.sh"]

