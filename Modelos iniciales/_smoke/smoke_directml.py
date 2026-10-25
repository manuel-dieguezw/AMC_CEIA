"""Smoke test temporal: compara tiempo/epoca de ULCNN en CPU vs DirectML (RX 6700).
Correr con el python de amc_env_dml (tiene torch_directml instalado). Subconjunto
chico del dataset real, pocas epocas, solo para medir velocidad.

Borrar esta carpeta (_smoke/) al terminar.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar "common"

import numpy as np
import torch

from common.data import load_radioml_digital, split_dataset, make_loader
from common.models import ULCNN
from common.train import train_model

DATASET_FOLDER = str(Path(__file__).resolve().parent.parent / "rfml")
N_SUBSET = 4000   # subconjunto chico, solo para medir velocidad
EPOCHS = 3


def run(device_name, device):
    print(f"\n=== Device: {device_name} ({device}) ===")
    X, y, snrs, _ = load_radioml_digital(dataset_folder=DATASET_FOLDER)
    X, y, snrs = X[:N_SUBSET], y[:N_SUBSET], snrs[:N_SUBSET]
    (X_train, y_train, _), (X_val, y_val, _), _ = split_dataset(X, y, snrs)

    train_loader = make_loader(X_train, y_train, batch_size=256)
    val_loader = make_loader(X_val, y_val, batch_size=256, shuffle=False)

    model = ULCNN()
    t0 = time.time()
    model, history = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, device=device, patience=EPOCHS + 1
    )
    elapsed = time.time() - t0
    print(f"{device_name}: {elapsed:.1f}s totales, {elapsed / EPOCHS:.1f}s/epoca "
          f"(val_acc final: {history['val_acc'][-1]:.3f})")
    return elapsed / EPOCHS


if __name__ == "__main__":
    # Importante: correr CPU y DirectML en el MISMO proceso rompe el motor de autograd
    # (bug conocido de torch-directml: el tamaño de device_ready_queues_ se fija con el
    # primer device usado). Elegir un solo device por invocación:
    #   python smoke_directml.py cpu
    #   python smoke_directml.py dml
    which = sys.argv[1] if len(sys.argv) > 1 else "cpu"

    if which == "cpu":
        run("CPU", "cpu")
    elif which == "dml":
        import torch_directml
        run("DirectML", torch_directml.device())
    else:
        raise SystemExit("uso: smoke_directml.py [cpu|dml]")
