print("[DEBUG] Etape 1 : Imports de base...")
import io
import platform
from pathlib import Path

print("[DEBUG] Etape 2 : Patch WMI & Platform...")
# Contournement : sur certaines machines Windows, les requetes WMI de platform.py
# (via uname(), win32_ver(), etc.) sont extremement lentes ou se bloquent,
# ce qui freeze l'import de torch (qui appelle platform.machine() au chargement).
# On mock platform.machine() et platform.uname() pour retourner instantanement
# des valeurs standard et eviter tout appel systeme bloquant.
platform.machine = lambda: "AMD64"

from collections import namedtuple
_uname_fields = ['system', 'node', 'release', 'version', 'machine', 'processor']
_uname_result = namedtuple('uname_result', _uname_fields)
platform.uname = lambda: _uname_result(
    system="Windows",
    node="localhost",
    release="10",
    version="10.0.19045",
    machine="AMD64",
    processor="Intel64"
)

# On mock aussi les autres raccourcis au cas ou
platform.system = lambda: "Windows"
platform.release = lambda: "10"
platform.version = lambda: "10.0.19045"

print("[DEBUG] Etape 3 : Import torch...")
import torch
print("[DEBUG] Etape 4 : Import torch.nn...")
import torch.nn as nn
print("[DEBUG] Etape 5 : Import torchvision...")
from torchvision import models, transforms
print("[DEBUG] Etape 6 : Import PIL...")
from PIL import Image
print("[DEBUG] Etape 7 : Import fastapi...")
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

print("[DEBUG] Etape 8 : Initialisation FastAPI...")
app = FastAPI(title="Detecteur de maladies de plantes")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

print("[DEBUG] Etape 9 : Configuration DEVICE...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] DEVICE selectionne : {DEVICE}")

def _find_model_path():
    for candidate in (BASE_DIR / "model" / "plant_model.pth", BASE_DIR / "model.pth"):
        if candidate.exists():
            return candidate
    return BASE_DIR / "model" / "plant_model.pth"

MODEL_PATH = _find_model_path()
print(f"[DEBUG] Chemin du modele trouve : {MODEL_PATH}")

CLASSES = ["Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy","Potato_Early_blight","Potato_healthy"]
CLASS_LABELS = {
    "Tomato_Bacterial_spot": "Tomate - Tache bacterienne",
    "Tomato_Early_blight":   "Tomate - Mildiou precoce",
    "Tomato_healthy":        "Tomate - Saine",
    "Potato_Early_blight":   "Pomme de terre - Mildiou precoce",
    "Potato_healthy":        "Pomme de terre - Saine",
}

def _build_head(in_features, num_classes, state_dict):
    # Detecte l'architecture de la tete a partir des cles du checkpoint :
    # tete simple (fc.weight) vs tete Sequential (fc.0.weight / fc.3.weight)
    if any(k.startswith("fc.0.") or k.startswith("fc.3.") for k in state_dict):
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    return nn.Linear(in_features, num_classes)

def load_model():
    print("[DEBUG] Etape 11 : Chargement de ResNet18...")
    classes, labels = CLASSES, CLASS_LABELS
    m = models.resnet18(weights=None)
    print(f"[DEBUG] ResNet18 instancie. Verification du fichier modele : {MODEL_PATH.exists()}")
    if MODEL_PATH.exists():
        print(f"[DEBUG] Lecture du checkpoint depuis {MODEL_PATH}...")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        print("[DEBUG] Checkpoint lu avec succes. Extraction de l'architecture...")
        state_dict = ckpt["model_state_dict"]
        classes = ckpt.get("classes") or ckpt.get("class_names") or CLASSES
        labels = ckpt.get("class_labels", CLASS_LABELS)
        print("[DEBUG] Adaptation de la tete de classification...")
        m.fc = _build_head(m.fc.in_features, len(classes), state_dict)
        print("[DEBUG] Chargement des poids du modele (load_state_dict)...")
        m.load_state_dict(state_dict)
        print("[DEBUG] Poids charges.")
    else:
        print("[DEBUG] Aucun fichier modele trouve. Initialisation de la tete par defaut...")
        m.fc = nn.Sequential(
            nn.Linear(m.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASSES)),
        )
    m.eval()
    print("[DEBUG] Modele pret et mis en mode eval().")
    return m.to(DEVICE), classes, labels

print("[DEBUG] Etape 10 : Lancement de load_model()...")
model, CLASSES, CLASS_LABELS = load_model()
print("[DEBUG] Etape 12 : Modele charge avec succes !")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img    = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx  = probs.argmax().item()
    pred = CLASSES[idx]
    return {
        "prediction": CLASS_LABELS.get(pred, pred),
        "confidence": round(probs[idx].item(), 4),
        "is_healthy": "healthy" in pred.lower(),
        "all_probs":  {CLASS_LABELS.get(c,c): round(probs[i].item(),4) for i,c in enumerate(CLASSES)},
    }

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
