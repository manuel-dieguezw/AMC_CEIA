"""Loop de entrenamiento y utilidades de evaluación compartidas por los
notebooks cnn/, lstm/, gru/, tcn/ y por comparacion/.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix

from .data import make_loader


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, min_epochs=10):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs  # épocas mínimas antes de activarse
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None
        self.stop = False

    def __call__(self, val_loss, model, epoch):
        # Siempre guardar los mejores pesos
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        # Solo activar el stop si ya pasaron las épocas mínimas
        if epoch >= self.min_epochs and self.counter >= self.patience:
            self.stop = True


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-3,
    device="cpu",
    patience=10,
    use_lr_scheduler=False,
    scheduler_factor=0.8,
    scheduler_patience=10,
    scheduler_min_lr=1e-7,
):
    """`use_lr_scheduler=False` (default) reproduce exactamente el loop original —
    no cambia el resultado de los notebooks existentes. Con `True`, agrega
    `ReduceLROnPlateau` sobre `val_loss` (usado por `cnn_preproc/`, cuya receta de
    entrenamiento original lo incluye)."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr
        )
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_losses.append(criterion(logits, y_batch).item())
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step(val_loss)

        early_stopping(val_loss, model, epoch)
        if (epoch + 1) % 10 == 0 or early_stopping.stop:
            print(
                f"Época {epoch + 1:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Patience: {early_stopping.counter}/{early_stopping.patience}"
            )
        if early_stopping.stop:
            print(f"Early stopping en época {epoch + 1}. Mejor val_loss: {early_stopping.best_loss:.4f}")
            break

    model.load_state_dict(early_stopping.best_weights)
    return model, history


def evaluate_by_snr(model, X_test, y_test, snr_test, device="cpu"):
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in sorted(np.unique(snr_test)):
            mask = snr_test == snr
            X_snr = torch.tensor(X_test[mask], dtype=torch.float32).to(device)
            y_snr = torch.tensor(y_test[mask], dtype=torch.long).to(device)
            preds = model(X_snr).argmax(dim=1)
            results[snr] = (preds == y_snr).float().mean().item()
    return results


def overall_accuracy(model, X_test, y_test, device="cpu", batch_size=512):
    model.eval()
    loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return correct / total


def compute_cm(model, X_test, y_test, device="cpu", batch_size=512):
    model.eval()
    loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
    return confusion_matrix(all_true, all_preds, normalize="true")
