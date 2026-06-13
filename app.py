# app.py — Point d'entrée principal pour Hugging Face Spaces et l'exécution locale
import sys
import subprocess

if __name__ == "__main__":
    # Hugging Face Spaces écoute par défaut sur le port 7860
    # En local, tu peux exécuter via ce script !
    print("Démarrage du serveur uvicorn...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0", "--port", "7860"
    ])
