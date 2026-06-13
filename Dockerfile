# Utiliser une image Python officielle légère
FROM python:3.12-slim

# Définir le dossier de travail
WORKDIR /code

# Installer les dépendances système requises
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement le fichier de dépendances d'abord pour optimiser le cache Docker
COPY requirements.txt .

# Installer les dépendances Python (version CPU de torch pour la légèreté sur HF)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Créer le dossier du modèle s'il n'existe pas
RUN mkdir -p model

# Copier le reste du code de l'application
COPY . .

# Donner les permissions de lecture/écriture au dossier de l'application (requis sur HF Spaces)
RUN chmod -R 777 /code

# Hugging Face Spaces écoute sur le port 7860
EXPOSE 7860

# Commande de démarrage direct via uvicorn (évite tout conflit d'app.py)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
