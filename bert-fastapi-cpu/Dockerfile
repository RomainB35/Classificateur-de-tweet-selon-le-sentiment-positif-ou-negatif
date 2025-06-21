FROM python:3.10-slim

# Installer les paquets système minimum nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Mettre pip à jour + installer torch CPU-only avant le reste
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.1.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt

# Copier le code de l’application
COPY app.py .
COPY saved_model/ ./saved_model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

