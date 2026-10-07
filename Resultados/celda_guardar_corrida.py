# --- Guardar esta corrida (no pisa nada: cada ejecución crea su propia carpeta) ---------------------
# Pegar como celda NUEVA DESPUÉS de la celda de evaluación de cada notebook. Usa las variables que el
# notebook ya tiene (model, history, acc, snr_acc, X_test, y_test, snr_test, MODEL_KWARGS, BATCH_SIZE,
# EPOCHS, PATIENCE, SEED). Guarda en Resultados/sweep/<id>/ con el mismo formato que Resultados/sweep.py,
# así `python Resultados/sweep.py --resumen` junta corridas de notebooks y del barrido en una tabla.
import json
from datetime import datetime
from pathlib import Path

LR = 1e-3        # <-- poner el lr que se usó en train_model(...) de ESTE notebook (lstm: 3e-4)
AUGMENT = True   # <-- False si se entrenó sin augment_rotate_iq
SCHEDULER = False  # <-- True si se usó use_lr_scheduler=True (cnn_preproc)
DATA_CONFIG = {"data_source": "radioml"}  # <-- si se entrenó con datos sintéticos: Datos_Sinteticos.generador.config(...)

_folder = globals().get("RUN_NAME", Path.cwd().name)  # cnn, lstm, gru, tcn, cnn_ap, tcn_ap, cnn_preproc
_ARCH = {"cnn": ("cnn", "iq"), "lstm": ("lstm", "iq"), "gru": ("gru", "iq"), "tcn": ("tcn", "iq"),
         "cnn_ap": ("cnn", "ap"), "tcn_ap": ("tcn", "ap"), "cnn_preproc": ("ulcnn", "iq")}
_arch, _repr = _ARCH.get(_folder, (_folder, "iq"))
_cur = bool(globals().get("USE_CURRICULUM", False))
_low, _pos = snr_test <= -10, snr_test > 0

_cfg = {
    "arch": _arch, "repr": _repr, "target": "notebook", "params": int(model.count_parameters()),
    "model_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in MODEL_KWARGS.items()},
    "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "patience": PATIENCE,
    "augment": AUGMENT, "curriculum": _cur, "lr_scheduler": SCHEDULER,
    "curriculum_kwargs": globals().get("CURRICULUM_KWARGS") if _cur else None,
    **DATA_CONFIG, "seed": SEED, "fuente": f"notebook {_folder}", "fecha": datetime.now().isoformat(timespec="seconds"),
}
_metrics = {
    "accuracy_global": float(acc),
    "accuracy_snr_menor_igual_-10": float(overall_accuracy(model, X_test[_low], y_test[_low], device=DEVICE)),
    "accuracy_snr_mayor_0": float(overall_accuracy(model, X_test[_pos], y_test[_pos], device=DEVICE)),
    "accuracy_por_snr": {str(int(k)): float(v) for k, v in snr_acc.items()},
    "parametros": _cfg["params"],
    "epocas_entrenadas": len(history["train_loss"]),
    "mejor_val_acc": float(max(history["val_acc"])),
    "mejor_val_loss": float(min(history["val_loss"])),
    "segundos_entrenamiento": 0.0,  # el notebook no mide el tiempo
}

_rid = (f"{_arch}_{_repr}_p{_cfg['params']}_lr{LR:g}_bs{BATCH_SIZE}_aug{int(AUGMENT)}_cur{int(_cur)}{'_sch' if SCHEDULER else ''}"
        f"_seed{SEED}_nb{datetime.now():%Y%m%d-%H%M%S}")
_out = Path(globals().get("SWEEP_DIR", Path.cwd().resolve().parents[1] / "Resultados" / "sweep")) / _rid
_out.mkdir(parents=True, exist_ok=False)  # falla en vez de pisar si ya existiera
(_out / "config.json").write_text(json.dumps(_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
(_out / "history.json").write_text(json.dumps(
    {k: [float(x) for x in v] for k, v in history.items() if isinstance(v, list)}), encoding="utf-8")
torch.save(model.state_dict(), _out / "model.pt")
(_out / "metrics.json").write_text(json.dumps(_metrics, indent=2), encoding="utf-8")  # marca de "terminada"
print(f"Corrida guardada en {_out}")
