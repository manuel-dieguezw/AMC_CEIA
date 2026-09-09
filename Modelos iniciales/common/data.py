"""Carga y preparación del dataset RadioML 2016.10a (subset digital, 8 modulaciones),
compartida por los notebooks de entrenamiento (cnn/, lstm/, gru/, tcn/) y por comparacion/.
"""
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATASET_FOLDER = "../rfml"


def load_radioml_digital(dataset_folder=DATASET_FOLDER, seed=42):
    X = np.load(f"{dataset_folder}/radioml_X_digital.npy")
    meta = pl.read_parquet(f"{dataset_folder}/radioml_metadata_digital.parquet")
    mods = meta["mod"].to_numpy()
    snrs = meta["snr"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(mods)

    # Mezclar antes del split (el dataset viene ordenado por clase)
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X, y, snrs = X[idx], y[idx], snrs[idx]

    return X, y, snrs, le


def split_dataset(X, y, snrs, seed=42):
    """Split 60/20/20 estratificado por clase."""
    X_train, X_temp, y_train, y_temp, snr_train, snr_temp = train_test_split(
        X, y, snrs, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test, snr_val, snr_test = train_test_split(
        X_temp, y_temp, snr_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return (X_train, y_train, snr_train), (X_val, y_val, snr_val), (X_test, y_test, snr_test)


def to_amplitude_phase(X):
    """Convierte señales I/Q (N, 2, L) a representación Amplitud/Fase (N, 2, L).

    Canal 0: amplitud = sqrt(I² + Q²). Canal 1: fase = atan2(Q, I), en radianes.
    Mismo shape que la entrada (2 canales), por lo que cualquier arquitectura
    de `common/models.py` pensada para I/Q acepta A/φ sin cambios.
    """
    I, Q = X[:, 0, :], X[:, 1, :]
    amplitude = np.sqrt(I ** 2 + Q ** 2)
    phase = np.arctan2(Q, I)
    return np.stack([amplitude, phase], axis=1).astype(X.dtype)


def augment_rotate_iq(X, y, snrs, angles=(np.pi / 2, np.pi, 3 * np.pi / 2)):
    """Data augmentation por rotación de la constelación I/Q (usada por ULCNN/`cnn_preproc`).

    Multiplica cada muestra (I, Q) por la matriz de rotación 2D de cada ángulo en `angles` y
    concatena las versiones rotadas al dataset original — con los 3 ángulos default, el
    dataset queda 4x más grande. Aplicar **solo al split de entrenamiento**: rotar val/test
    invalidaría la evaluación (dejaría de medir sobre las señales reales del split).
    """
    I, Q = X[:, 0, :], X[:, 1, :]
    partes_X, partes_y, partes_snrs = [X], [y], [snrs]
    for theta in angles:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        I_rot = cos_t * I - sin_t * Q
        Q_rot = sin_t * I + cos_t * Q
        partes_X.append(np.stack([I_rot, Q_rot], axis=1).astype(X.dtype))
        partes_y.append(y)
        partes_snrs.append(snrs)
    return np.concatenate(partes_X, axis=0), np.concatenate(partes_y, axis=0), np.concatenate(partes_snrs, axis=0)


def make_loader(X, y, batch_size=256, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)
