"""Loop de entrenamiento y utilidades de evaluación compartidas por los
notebooks cnn/, lstm/, gru/, tcn/ y por comparacion/.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import confusion_matrix

from .data import make_loader


class SNRCurriculum:
    """Entrenamiento por grupos de SNR, de mayor a menor, con tres criterios de avance:

    - mode="acc"  (default): val accuracy sobre todos los SNR controla el early stopping
      dentro de cada etapa y decide si se suman más etapas. Comportamiento original.
    - mode="stage_val": dentro de cada etapa, el early stopping se basa solo en las muestras
      de val cuyo SNR pertenece al rango visible en esa etapa. El checkpoint global se elige
      por la mejor val accuracy sobre el set completo. Esto evita que el ruido de los SNR bajos
      no vistos contamine el criterio de parada en las etapas iniciales.
    - mode="fixed": N épocas fijas por grupo de SNR, sin mirar accuracy para decidir si avanzar
      ni cuándo cortar. El checkpoint global se elige por la mejor val accuracy (full set).

    Arranque (solo mode="acc"): mientras la mejor accuracy no supere el azar + `warmup_margin`,
    no se cuenta "sin progreso". Si tras `max_warmup_epochs` épocas sigue en el azar, se corta.

    Se usa pasándolo a `train_model(..., curriculum=...)`; el `train_loader` y el `patience`
    que recibe `train_model` se ignoran en ese caso.
    """

    def __init__(self, X_train, y_train, snr_train, batch_size=256, group_size=4,
                 patience=3, min_gain=0.005, max_epochs_per_stage=15,
                 warmup_margin=0.05, max_warmup_epochs=30,
                 mode="acc", epochs_per_stage=10,
                 X_val=None, y_val=None, snr_val=None, batch_size_val=512):
        self.snr = np.asarray(snr_train)
        self.chance = 1.0 / len(np.unique(y_train))
        self.warmup_margin = warmup_margin
        self.max_warmup_epochs = max_warmup_epochs
        self.levels = sorted(np.unique(self.snr), reverse=True)  # de SNR alto a bajo
        self.dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.patience = patience
        self.min_gain = min_gain
        self.max_epochs_per_stage = max_epochs_per_stage
        self.n_stages = -(-len(self.levels) // group_size)  # ceil
        if mode not in ("acc", "stage_val", "fixed"):
            raise ValueError(f"mode debe ser 'acc', 'stage_val' o 'fixed', no {mode!r}")
        self.mode = mode
        self.epochs_per_stage = epochs_per_stage  # solo para mode="fixed"

        # Para mode="stage_val": val dataset filtrado por SNR dentro de cada etapa
        if mode == "stage_val":
            if X_val is None or y_val is None or snr_val is None:
                raise ValueError("mode='stage_val' requiere X_val, y_val y snr_val")
            self._snr_val = np.asarray(snr_val)
            self._val_dataset = TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)),
                torch.from_numpy(np.asarray(y_val, dtype=np.int64)),
            )
            self._batch_size_val = batch_size_val

    def stage_min_snr(self, stage):
        """SNR más bajo incluido en la etapa `stage` (0-based)."""
        return self.levels[min((stage + 1) * self.group_size, len(self.levels)) - 1]

    def loader(self, stage):
        idx = np.flatnonzero(self.snr >= self.stage_min_snr(stage))
        return DataLoader(Subset(self.dataset, idx.tolist()), batch_size=self.batch_size, shuffle=True)

    def stage_val_loader(self, stage):
        """Val loader con solo las muestras cuyo SNR cae en el rango visible de esta etapa (mode='stage_val')."""
        idx = np.flatnonzero(self._snr_val >= self.stage_min_snr(stage))
        return DataLoader(Subset(self._val_dataset, idx.tolist()), batch_size=self._batch_size_val, shuffle=False)


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


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def _validate(model, val_loader, criterion, device):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            losses.append(criterion(logits, y_batch).item())
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
    return np.mean(losses), correct / total


def _train_with_curriculum(model, val_loader, optimizer, criterion, cur, epochs, device, scheduler=None):
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "curriculum_min_snr": [], "curriculum_stage": []}
    best_acc = 0.0
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    total_epochs = 0

    for stage in range(cur.n_stages):
        loader = cur.loader(stage)
        min_snr = cur.stage_min_snr(stage)

        # ── mode="fixed": N épocas fijas por etapa, sin criterio de accuracy ──────────────────
        if cur.mode == "fixed":
            print(f"[curriculum fixed] etapa {stage + 1}/{cur.n_stages}: SNR >= {min_snr} dB "
                  f"({len(loader.dataset)} muestras, {cur.epochs_per_stage} épocas fijas)")
            for _ in range(cur.epochs_per_stage):
                if total_epochs >= epochs:
                    break
                train_loss = _train_one_epoch(model, loader, optimizer, criterion, device)
                val_loss, val_acc = _validate(model, val_loader, criterion, device)
                total_epochs += 1
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["curriculum_min_snr"].append(float(min_snr))
                history["curriculum_stage"].append(stage)
                if scheduler is not None:
                    scheduler.step(val_loss)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"Época {total_epochs:3d} | SNR >= {min_snr:>3} dB | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} (mejor {best_acc:.4f})")

        # ── mode="stage_val": early stopping solo sobre el rango de SNR visible ───────────────
        elif cur.mode == "stage_val":
            sv_loader = cur.stage_val_loader(stage)
            print(f"[curriculum stage_val] etapa {stage + 1}/{cur.n_stages}: SNR >= {min_snr} dB "
                  f"({len(loader.dataset)} train | {len(sv_loader.dataset)} val en rango)")
            no_improve, epochs_in_stage = 0, 0
            stage_best_acc = 0.0
            stage_best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            while epochs_in_stage < cur.max_epochs_per_stage and no_improve < cur.patience and total_epochs < epochs:
                train_loss = _train_one_epoch(model, loader, optimizer, criterion, device)
                _, stage_val_acc = _validate(model, sv_loader, criterion, device)   # para stopping
                val_loss, val_acc = _validate(model, val_loader, criterion, device)  # para checkpoint global
                total_epochs += 1
                epochs_in_stage += 1
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["curriculum_min_snr"].append(float(min_snr))
                history["curriculum_stage"].append(stage)
                if scheduler is not None:
                    scheduler.step(val_loss)
                # dentro de la etapa: stopping por stage_val_acc (SNR del rango visible)
                if stage_val_acc > stage_best_acc + cur.min_gain:
                    stage_best_acc = stage_val_acc
                    stage_best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                # checkpoint global: mejor full val_acc a lo largo de todo el entrenamiento
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"Época {total_epochs:3d} | SNR >= {min_snr:>3} dB | Train: {train_loss:.4f} | "
                      f"Val(full): {val_acc:.4f} | Val(stage): {stage_val_acc:.4f} | "
                      f"Mejor full: {best_acc:.4f} | Patience: {no_improve}/{cur.patience}")
            # siguiente etapa arranca desde los mejores pesos de esta etapa (por stage_val_acc)
            model.load_state_dict(stage_best_weights)

        # ── mode="acc": comportamiento original ───────────────────────────────────────────────
        else:
            print(f"[curriculum] etapa {stage + 1}/{cur.n_stages}: SNR >= {min_snr} dB ({len(loader.dataset)} muestras)")
            no_improve, epochs_in_stage, improved = 0, 0, False
            start_acc = cur.chance + cur.warmup_margin

            while (epochs_in_stage < cur.max_epochs_per_stage and no_improve < cur.patience
                   and total_epochs < epochs
                   and not (best_acc <= start_acc and total_epochs >= cur.max_warmup_epochs)):
                train_loss = _train_one_epoch(model, loader, optimizer, criterion, device)
                val_loss, val_acc = _validate(model, val_loader, criterion, device)
                total_epochs += 1
                epochs_in_stage += 1
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["curriculum_min_snr"].append(float(min_snr))
                history["curriculum_stage"].append(stage)
                if scheduler is not None:
                    scheduler.step(val_loss)
                if val_acc > best_acc + cur.min_gain:
                    best_acc, improved, no_improve = val_acc, True, 0
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                if best_acc <= start_acc:
                    no_improve, epochs_in_stage = 0, 0
                print(f"Época {total_epochs:3d} | SNR >= {min_snr:>3} dB | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} (mejor {best_acc:.4f})")

            model.load_state_dict(best_weights)
            if best_acc <= start_acc:
                print(f"AVISO: el modelo no salió del azar ({cur.chance:.3f}) tras {total_epochs} épocas — "
                      f"revisá lr / normalización de la entrada / arquitectura.")
                break
            if not improved:
                print(f"[curriculum] sumar SNR >= {min_snr} dB no mejoró la val accuracy (mejor {best_acc:.4f}): "
                      f"se corta acá y se conservan los mejores pesos.")
                break
            if total_epochs >= epochs:
                break

    model.load_state_dict(best_weights)
    return model, history


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
    curriculum=None,
    train_loader_fn=None,
):
    """`use_lr_scheduler=False` (default) reproduce exactamente el loop original —
    no cambia el resultado de los notebooks existentes. Con `True`, agrega
    `ReduceLROnPlateau` sobre `val_loss` (usado por `cnn_preproc/`, cuya receta de
    entrenamiento original lo incluye).

    `curriculum=None` (default) tampoco cambia nada. Con un `SNRCurriculum` se entrena por
    grupos de SNR de mayor a menor (ver esa clase): se ignoran `train_loader` y `patience`,
    y `epochs` pasa a ser un tope global de seguridad. El curriculum decide QUÉ datos entran
    y `use_lr_scheduler` decide el learning rate, así que se pueden combinar: el scheduler
    avanza una vez por época sobre `val_loss`, igual que sin curriculum.
    `history["curriculum_min_snr"]` guarda el SNR mínimo incluido en cada época.
    El modo del curriculum (mode="acc"/"stage_val"/"fixed") se configura al construir el
    SNRCurriculum; train_model no necesita saber cuál es.

    `train_loader_fn=None` (default) tampoco cambia nada: se entrena siempre con
    `train_loader`. Si se pasa, se llama **antes de cada época** (sin argumentos) y su
    resultado reemplaza a `train_loader` para esa época — pensado para datos sintéticos
    generados on-the-fly (un lote nuevo por época en vez de un lote fijo reusado), sin
    tocar el resto del loop. Se ignora si `curriculum` no es `None`."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr
        )

    if curriculum is not None:
        return _train_with_curriculum(model, val_loader, optimizer, criterion, curriculum, epochs, device,
                                      scheduler=scheduler)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        loader = train_loader_fn() if train_loader_fn is not None else train_loader
        train_loss = _train_one_epoch(model, loader, optimizer, criterion, device)
        val_loss, val_acc = _validate(model, val_loader, criterion, device)
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
