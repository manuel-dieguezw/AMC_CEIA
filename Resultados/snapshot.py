"""Snapshot de una etapa de experimentos.

Copia .pt / history.pkl / .png / notebook de cada modelo a Resultados/<etapa>/<grupo>/<carpeta>/ y
calcula Resultados/<etapa>/metricas.json (accuracy global, por SNR, a SNR <= -10 dB, parametros)
evaluando cada modelo en el test set original.

Uso (desde cualquier directorio, con el python del venv del proyecto):

    python Resultados/snapshot.py etapa_1_augmentation --grupos "Modelos iniciales"
    python Resultados/snapshot.py etapa_0_baseline --notebooks-from-git-head

--grupos      Solo snapshotea esos grupos (default: todos). Util para no copiar de nuevo modelos
              que no reentrenaste en esa etapa.
--notebooks-from-git-head
              Para "Modelos iniciales/", toma el notebook del ultimo commit en vez del actual
              (sirve para congelar una etapa cuyos notebooks ya editaste sin commitear).

IMPORTANTE - modelos nuevos: el script NO descubre arquitecturas solo. Cada modelo se declara en
SPECS (clase, nombre del .pt, representacion de entrada). Si agregas una carpeta nueva con
.pt + history.pkl y no la registras aca, el script te avisa al final con "SIN SPEC" en vez de
ignorarla en silencio.
"""
import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MI = ROOT / "Modelos iniciales"
MB = ROOT / "Modelos_Bajo_SNR"
sys.path.insert(0, str(MI))
sys.path.insert(0, str(MB))

from common.data import load_radioml_digital, split_dataset, to_amplitude_phase
from common.models import CNNBaseline, LSTMClassifier, GRUClassifier, TCN, ULCNN
from common.train import evaluate_by_snr, overall_accuracy
from common_snr.features import compute_hos_features, compute_stft, compute_cwt, pack_fusion_features
from common_snr.models import HOSClassifier, STFTCNN, SpectrogramULCNN, CNNLSTMHybrid, FusionClassifier

# (grupo, carpeta, clase, nombre_pt, representacion) — representacion: iq | ap | hos | stft | wavelet | fusion
# Para agregar un modelo nuevo: importar su clase arriba y sumar una linea aca.
SPECS = [
    ("Modelos iniciales", "cnn", CNNBaseline, "cnn.pt", "iq"),
    ("Modelos iniciales", "lstm", LSTMClassifier, "lstm.pt", "iq"),
    ("Modelos iniciales", "gru", GRUClassifier, "gru.pt", "iq"),
    ("Modelos iniciales", "tcn", TCN, "tcn.pt", "iq"),
    ("Modelos iniciales", "cnn_ap", CNNBaseline, "cnn_ap.pt", "ap"),
    ("Modelos iniciales", "tcn_ap", TCN, "tcn_ap.pt", "ap"),
    ("Modelos iniciales", "cnn_preproc", ULCNN, "cnn_preproc.pt", "iq"),
    ("Modelos_Bajo_SNR", "hos", HOSClassifier, "hos.pt", "hos"),
    ("Modelos_Bajo_SNR", "stft", STFTCNN, "stft.pt", "stft"),
    ("Modelos_Bajo_SNR", "wavelet", STFTCNN, "wavelet.pt", "wavelet"),
    ("Modelos_Bajo_SNR", "lstm_snr", CNNLSTMHybrid, "lstm_snr.pt", "iq"),
    ("Modelos_Bajo_SNR", "stft_ulcnn", SpectrogramULCNN, "stft_ulcnn.pt", "stft"),
    ("Modelos_Bajo_SNR", "fusion", FusionClassifier, "fusion.pt", "fusion"),
]
GRUPOS = sorted({s[0] for s in SPECS})
# rfml/ es el baseline con la libreria externa (sin history.pkl ni clases de common/): no se snapshotea.
IGNORAR = {"common", "common_snr", "comparacion", "rfml", "__pycache__"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("etapa")
parser.add_argument("--grupos", nargs="+", choices=GRUPOS, default=GRUPOS)
parser.add_argument("--notebooks-from-git-head", action="store_true")
parser.add_argument("--sobrescribir", action="store_true",
                    help="Permite pisar una etapa que ya existe (por defecto el script aborta).")
args = parser.parse_args()

# Guarda anti-pisado: se chequea antes de cargar datos o copiar nada.
_destino = ROOT / "Resultados" / args.etapa
if _destino.exists() and any(_destino.iterdir()) and not args.sobrescribir:
    sys.exit(f"ABORTADO: {_destino} ya existe y tiene contenido. No se toco nada.\n"
             f"Usa otro nombre de etapa, o --sobrescribir si de verdad queres pisarla.")

specs = [s for s in SPECS if s[0] in args.grupos]

X, y, snrs, le = load_radioml_digital(dataset_folder=str(MI / "rfml"), seed=42)
_, _, (X_test, y_test, snr_test) = split_dataset(X, y, snrs, seed=42)
n_classes = len(le.classes_)

reps = {}


def get_rep(name):
    if name in reps:
        return reps[name]
    if name == "iq":
        reps[name] = X_test
    elif name == "ap":
        reps[name] = to_amplitude_phase(X_test)
    elif name == "hos":
        reps[name] = compute_hos_features(X_test)
    elif name == "stft":
        reps[name] = compute_stft(X_test)
    elif name == "wavelet":
        print("calculando CWT del test set (~1 min)...")
        reps[name] = compute_cwt(X_test)
    elif name == "fusion":
        reps[name] = pack_fusion_features(X_test, get_rep("hos"), get_rep("stft"))
    else:
        raise ValueError(f"representacion desconocida: {name}")
    return reps[name]


out_root = ROOT / "Resultados" / args.etapa
metricas = {}

for grupo, carpeta, cls, pt_name, rep in specs:
    src = ROOT / grupo / carpeta
    dst = out_root / grupo / carpeta
    if not (src / pt_name).exists() or not (src / "history.pkl").exists():
        print(f"SKIP {grupo}/{carpeta}: falta {pt_name} o history.pkl")
        continue
    dst.mkdir(parents=True, exist_ok=True)

    for f in list(src.glob("*.pt")) + list(src.glob("*.pkl")) + list(src.glob("*.png")):
        shutil.copy2(f, dst / f.name)

    nb_name = f"{carpeta}.ipynb"
    if args.notebooks_from_git_head and grupo == "Modelos iniciales":
        res = subprocess.run(["git", "show", f"HEAD:{grupo}/{carpeta}/{nb_name}"], cwd=ROOT, capture_output=True)
        if res.returncode == 0:
            (dst / nb_name).write_bytes(res.stdout)
    elif (src / nb_name).exists():
        shutil.copy2(src / nb_name, dst / nb_name)

    with open(src / "history.pkl", "rb") as fh:
        history = pickle.load(fh)
    kwargs = history.get("model_kwargs", {})
    model = cls(n_classes=n_classes, **kwargs)
    model.load_state_dict(torch.load(src / pt_name, map_location="cpu"))
    model.eval()

    X_eval = get_rep(rep)
    acc = overall_accuracy(model, X_eval, y_test, "cpu")
    by_snr = evaluate_by_snr(model, X_eval, y_test, snr_test, "cpu")
    acc_bajo = overall_accuracy(model, X_eval[snr_test <= -10], y_test[snr_test <= -10], "cpu")
    key = f"{grupo}/{carpeta}"
    metricas[key] = {
        "clase": cls.__name__,
        "representacion": rep,
        "parametros": int(model.count_parameters()),
        "accuracy_global": float(acc),
        "accuracy_snr_menor_igual_-10": float(acc_bajo),
        "accuracy_por_snr": {str(int(k)): float(v) for k, v in by_snr.items()},
        "epocas_entrenadas": len(history["train_loss"]),
        "mejor_val_loss": float(min(history["val_loss"])),
        "model_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
    }
    print(f"OK {key}: params={metricas[key]['parametros']:,} acc={acc:.4f} acc(<=-10dB)={acc_bajo:.4f}")

out_root.mkdir(parents=True, exist_ok=True)
with open(out_root / "metricas.json", "w", encoding="utf-8") as fh:
    json.dump({"etapa": args.etapa, "modelos": metricas}, fh, indent=2, ensure_ascii=False)
print("metricas.json escrito en", out_root)

# Aviso de modelos no registrados: carpetas con .pt + history.pkl que SPECS no conoce.
registrados = {(s[0], s[1]) for s in SPECS}
for grupo in args.grupos:
    for carpeta in sorted((ROOT / grupo).iterdir()):
        if not carpeta.is_dir() or carpeta.name in IGNORAR:
            continue
        if (grupo, carpeta.name) not in registrados and list(carpeta.glob("*.pt")) and (carpeta / "history.pkl").exists():
            print(f"SIN SPEC: {grupo}/{carpeta.name} tiene .pt + history.pkl pero no esta en SPECS de snapshot.py "
                  f"-> NO se snapshoteo. Agregalo a SPECS.")
