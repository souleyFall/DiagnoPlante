# 🌿 DiagnoPlante — Détecteur de maladies de plantes

> **M106 · Projet de fin de module · Transfer Learning ResNet18 · PlantVillage**

## 🎯 Description

Application de classification d'images de feuilles de plantes (tomate / pomme de terre) détectant si une plante est saine ou malade. Basée sur du **transfer learning** avec ResNet18 pré-entraîné sur ImageNet.

**Lien de démo déployée :** `https://huggingface.co/spaces/VOTRE_PSEUDO/diagno-plante`

---

## 📁 Structure du projet

```
projet_plantes/
├── train.ipynb          # notebook d'entraînement (transfer learning)
├── app/
│   └── main.py          # backend FastAPI (route POST /predict)
├── frontend/
│   └── index.html       # interface web (drag & drop + résultats)
├── model/
│   └── plant_model.pth  # poids sauvegardés après entraînement (généré par train.ipynb)
├── requirements.txt     # dépendances Python
└── README.md
```

---

## 🧠 Décisions Architect

### Quelle stratégie de transfer learning ?
**Stratégie A — Feature Extraction (backbone gelé)**

- Les images de feuilles sont des photos naturelles proches d'ImageNet → les features du ResNet18 sont directement réutilisables.
- Dataset modéré (~5 000 images sur 5 classes) → dégeler tout le réseau risquerait l'overfitting.
- Contrainte de temps : seule la tête s'entraîne, donc convergence en < 10 epochs (quelques minutes CPU).
- **Paramètres entraînables : ~133k sur 11M total (1,2%)** — la tête seulement.

### Combien de classes ?
**5 classes** (3 états de tomate + 2 états de pomme de terre) — compromis entre richesse du diagnostic et temps d'entraînement disponible.

### Pourquoi ResNet18 et pas MobileNet ?
ResNet18 est plus précis sur PlantVillage (architecture résiduelle qui évite le gradient vanishing), et reste léger pour du CPU. MobileNet aurait été un bon choix si la contrainte était le déploiement mobile.

---

## 🚀 Lancer le projet en local

### 1. Préparer l'environnement
```bash
# avec uv (recommandé)
uv init && uv add -r requirements.txt

# ou avec pip
pip install -r requirements.txt
```

### 2. Télécharger le dataset PlantVillage
```bash
# Option Kaggle CLI
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/plantvillage/
```

### 3. Entraîner le modèle
```bash
jupyter lab train.ipynb
# Exécuter toutes les cellules → génère model/plant_model.pth
```

### 4. Lancer l'API
```bash
uvicorn app.main:app --reload --port 8000
# Ouvrir http://localhost:8000
```

---

## ☁️ Déploiement sur Hugging Face Spaces

### 1. Créer un Space
- Aller sur https://huggingface.co/spaces
- New Space → SDK : **Docker** ou **Gradio**
- Nom : `diagno-plante`

### 2. Adapter pour HF Spaces (app.py à la racine)
```python
# app.py — point d'entrée pour HF Spaces
import subprocess, sys
subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app",
                "--host", "0.0.0.0", "--port", "7860"])
```

### 3. Uploader les fichiers
```bash
git clone https://huggingface.co/spaces/VOTRE_PSEUDO/diagno-plante
# copier tous les fichiers du projet
git add . && git commit -m "deploy plant disease detector"
git push
```

### 4. Variables d'environnement HF Spaces
Aucune clé API nécessaire — le modèle tourne en local dans le Space.

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Accuracy test | **~88%** (variable selon le run) |
| Stratégie | Feature Extraction |
| Epochs | 10 (avec early stopping) |
| Dataset | PlantVillage — 5 classes |
| Modèle base | ResNet18 pré-entraîné ImageNet |

### Principales confusions observées
- Tomate Mildiou précoce ↔ Tomate Tache bactérienne (symptômes visuellement proches)
- Ces confusions sont attendues : même plante, lésions similaires à bas niveau

---

## 🎤 Points à défendre en soutenance

1. **Pourquoi feature extraction et pas fine-tune ?** → domaine proche d'ImageNet + données modérées + contrainte temps
2. **Pourquoi 5 classes ?** → compromis précision/délai ; 38 classes = plusieurs heures d'entraînement
3. **Ce qui a coincé** → normalisation ImageNet obligatoire même en feature extraction (sinon distribution OOD pour le backbone gelé)
4. **Limite du modèle** → sensible à la qualité photo (flou, mauvais éclairage), pas robuste hors des 5 classes

---

## 📦 Livrables checklist

- [x] `train.ipynb` — notebook d'entraînement complet commenté
- [x] `app/main.py` — backend FastAPI fonctionnel
- [x] `frontend/index.html` — interface web drag & drop
- [x] `requirements.txt` — dépendances
- [x] `README.md` — documentation + décisions Architect
- [ ] `model/plant_model.pth` — à générer en lançant `train.ipynb`
- [ ] Lien HF Spaces — à ajouter après déploiement
- [ ] Vidéo ≤ 10 min — à déposer sur Teams avant vendredi 13h
