"""Barrido de hiperparámetros y capacidad, independiente de los notebooks.

No toca ningún notebook ni ninguna etapa de Resultados/. Cada corrida se guarda en su propia
carpeta Resultados/sweep/<run_id>/ con TODO lo necesario para reproducirla y compararla:

    config.json    (incluye data_source: "radioml" o "synthetic" + parametros del generador) arquitectura, kwargs del modelo, parametros, lr, batch_size, epochs, patience,
                   augmentation, curriculum, semilla, representacion, subsets, fecha
    metrics.json   accuracy global / por SNR / a SNR <= -10 dB / SNR > 0, epocas, tiempo, mejor val
                   (se escribe AL FINAL: si falta, la corrida quedo incompleta y se reintenta)
    history.json   curvas de train/val por epoca
    model.pt       pesos

Una corrida cuyo metrics.json ya existe se SALTEA: nunca se pisa un resultado. Se puede cortar el
script y volver a lanzarlo, retoma donde quedó.

Comparar "a igual cantidad de parametros" (lo que pidio el tutor): --params acepta metas de
cantidad de parametros (p. ej. 10000 50000) y para cada arquitectura busca el ancho cuyo conteo
real se acerque más; "base" usa la configuración actual de los notebooks. Así se puede ver, por
ejemplo, cómo rinde ULCNN con la capacidad de TCN, o TCN con la de ULCNN.

Uso (con el python del venv del proyecto):

    # ver el plan sin entrenar nada (cantidad real de parametros de cada corrida y cuales ya existen)
    python Resultados/sweep.py --archs cnn tcn ulcnn --params base 10000 50000 --dry-run

    # correr
    python Resultados/sweep.py --archs cnn tcn gru lstm ulcnn --params 10000 50000 \\
        --lrs 1e-3 3e-4 --batch-sizes 128 256 --augment 1 --curriculum 0 1

    # tabla + grafico accuracy vs parametros con todo lo que haya en Resultados/sweep/
    python Resultados/sweep.py --resumen
"""
import argparse
import itertools
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MI = ROOT / "Modelos iniciales"
sys.path.insert(0, str(MI))

from common.data import load_radioml_digital, split_dataset, make_loader, to_amplitude_phase, augment_rotate_iq
from common.models import CNNBaseline, LSTMClassifier, GRUClassifier, TCN, ULCNN
from common.train import train_model, SNRCurriculum, evaluate_by_snr, overall_accuracy

N_CLASSES = 8
SEED_SPLIT = 42  # el split train/val/test es siempre el mismo que en los notebooks

# Para cada arquitectura: clase, perilla de "ancho" que se barre para llegar a una meta de parametros
# (`make(w)` devuelve los kwargs), la config actual de los notebooks (`base`) y el lr por defecto.
ARCHS = {
    "cnn": dict(cls=CNNBaseline, widths=range(2, 257, 2), lr=1e-3,
                make=lambda w: dict(channels=(w, w), kernel_size=3, fc_hidden=max(8, w), dropout=0.5),
                base=dict(channels=(64, 64), kernel_size=3, fc_hidden=128, dropout=0.5)),
    "lstm": dict(cls=LSTMClassifier, widths=range(4, 513, 2), lr=3e-4,
                 make=lambda w: dict(hidden_size=w, num_layers=2, fc_hidden=max(8, w // 2), dropout=0.5, input_norm=True),
                 base=dict(hidden_size=128, num_layers=2, fc_hidden=64, dropout=0.5, input_norm=True)),
    "gru": dict(cls=GRUClassifier, widths=range(4, 513, 2), lr=1e-3,
                make=lambda w: dict(hidden_size=w, num_layers=2, fc_hidden=max(8, w // 2), dropout=0.4, input_norm=True),
                base=dict(hidden_size=64, num_layers=2, fc_hidden=32, dropout=0.4, input_norm=True)),
    "tcn": dict(cls=TCN, widths=range(2, 257, 1), lr=1e-3,
                make=lambda w: dict(channels=w, dilations=(1, 2, 4), dropout=0.4),
                base=dict(channels=64, dilations=(1, 2, 4), dropout=0.4)),
    "ulcnn": dict(cls=ULCNN, widths=range(1, 129, 1), lr=1e-3,
                  make=lambda w: dict(n_cv=w, kernel_size=5, n_fmdr=6, shuffle_groups=2, attn_reduction=2),
                  base=dict(n_cv=16, kernel_size=5, n_fmdr=6, shuffle_groups=2, attn_reduction=2)),
}

_width_cache = {}


def count_params(arch, kwargs):
    return ARCHS[arch]["cls"](n_classes=N_CLASSES, **kwargs).count_parameters()


def kwargs_for_target(arch, target):
    """`target` = "base" o cantidad de parametros deseada. Devuelve (kwargs, parametros_reales)."""
    spec = ARCHS[arch]
    if target == "base":
        kw = spec["base"]
        return kw, count_params(arch, kw)
    if arch not in _width_cache:
        _width_cache[arch] = [(w, count_params(arch, spec["make"](w))) for w in spec["widths"]]
    w, n = min(_width_cache[arch], key=lambda wn: abs(wn[1] - int(target)))
    return spec["make"](w), n


def jsonable(kw):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}


def run_id(cfg):
    lr = f"{cfg['lr']:g}"
    return (f"{cfg['arch']}_{cfg['repr']}_p{cfg['params']}_lr{lr}_bs{cfg['batch_size']}"
            f"_aug{int(cfg['augment'])}_cur{int(cfg['curriculum'])}{'_sch' if cfg.get('lr_scheduler') else ''}"
            f"_seed{cfg['seed']}")


def build_runs(args):
    runs = []
    for arch, target, repr_, bs, aug, cur, sch, seed in itertools.product(
            args.archs, args.params, args.repr, args.batch_sizes, args.augment, args.curriculum, args.scheduler, args.seeds):
        kw, n = kwargs_for_target(arch, target)
        lrs = [ARCHS[arch]["lr"]] if args.lrs == ["auto"] else [float(x) for x in args.lrs]
        for lr in lrs:
            epochs = max(args.epochs, 80) if cur else args.epochs
            runs.append(dict(arch=arch, repr=repr_, target=target, params=n, model_kwargs=jsonable(kw), lr=lr,
                             batch_size=bs, epochs=epochs, patience=args.patience, augment=bool(aug),
                             curriculum=bool(cur), lr_scheduler=bool(sch), seed=seed, data_source="radioml", train_subset=args.train_subset,
                             val_subset=args.val_subset))
    unicas = {}
    for r in runs:
        unicas.setdefault(run_id(r), r)
    return list(unicas.values())


def prepare_data(cfg, split):
    (Xtr, ytr, str_), (Xva, yva, _), (Xte, yte, ste) = split
    if cfg["augment"]:  # solo train; antes de pasar a A/phi porque la rotacion es sobre I/Q
        Xtr, ytr, str_ = augment_rotate_iq(Xtr, ytr, str_)
    if cfg["repr"] == "ap":
        Xtr, Xva, Xte = to_amplitude_phase(Xtr), to_amplitude_phase(Xva), to_amplitude_phase(Xte)
    rng = np.random.default_rng(0)
    if cfg["train_subset"]:
        i = rng.choice(len(Xtr), min(cfg["train_subset"], len(Xtr)), replace=False)
        Xtr, ytr, str_ = Xtr[i], ytr[i], str_[i]
    if cfg["val_subset"]:
        i = rng.choice(len(Xva), min(cfg["val_subset"], len(Xva)), replace=False)
        Xva, yva = Xva[i], yva[i]
    return (Xtr, ytr, str_), (Xva, yva), (Xte, yte, ste)


def run_one(cfg, split, out_dir):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    (Xtr, ytr, str_), (Xva, yva), (Xte, yte, ste) = prepare_data(cfg, split)

    model = ARCHS[cfg["arch"]]["cls"](n_classes=N_CLASSES, **{k: tuple(v) if isinstance(v, list) else v
                                                              for k, v in cfg["model_kwargs"].items()})
    train_loader = make_loader(Xtr, ytr, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = make_loader(Xva, yva, batch_size=512, shuffle=False)
    curriculum = SNRCurriculum(Xtr, ytr, str_, batch_size=cfg["batch_size"]) if cfg["curriculum"] else None

    t0 = time.time()
    model, history = train_model(model, train_loader, val_loader, epochs=cfg["epochs"], lr=cfg["lr"],
                                 patience=cfg["patience"], curriculum=curriculum,
                                 use_lr_scheduler=cfg.get("lr_scheduler", False),
                                 scheduler_factor=0.8, scheduler_patience=10, scheduler_min_lr=1e-7)
    train_seconds = time.time() - t0

    by_snr = evaluate_by_snr(model, Xte, yte, ste)
    low, pos = ste <= -10, ste > 0
    metrics = {
        "accuracy_global": overall_accuracy(model, Xte, yte),
        "accuracy_snr_menor_igual_-10": overall_accuracy(model, Xte[low], yte[low]),
        "accuracy_snr_mayor_0": overall_accuracy(model, Xte[pos], yte[pos]),
        "accuracy_por_snr": {str(int(k)): float(v) for k, v in by_snr.items()},
        "parametros": int(model.count_parameters()),
        "epocas_entrenadas": len(history["train_loss"]),
        "mejor_val_acc": float(max(history["val_acc"])),
        "mejor_val_loss": float(min(history["val_loss"])),
        "segundos_entrenamiento": round(train_seconds, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(dict(cfg, fecha=datetime.now().isoformat(timespec="seconds"),
                                                         torch=torch.__version__, python=platform.python_version()),
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in history.items()}), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # marca de "terminada"
    return metrics


def resumen(out_root):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for m in sorted(out_root.glob("*/metrics.json")):
        cfg = json.loads((m.parent / "config.json").read_text(encoding="utf-8"))
        met = json.loads(m.read_text(encoding="utf-8"))
        rows.append(dict(run=m.parent.name, arch=cfg["arch"], repr=cfg["repr"], params=met["parametros"], lr=cfg["lr"],
                         batch_size=cfg["batch_size"], augment=cfg["augment"], curriculum=cfg["curriculum"],
                         seed=cfg["seed"], epocas=met["epocas_entrenadas"], acc=met["accuracy_global"],
                         acc_snr_le_m10=met["accuracy_snr_menor_igual_-10"], acc_snr_gt0=met["accuracy_snr_mayor_0"],
                         min=round(met["segundos_entrenamiento"] / 60, 1), model_kwargs=json.dumps(cfg["model_kwargs"])))
    if not rows:
        print(f"No hay corridas terminadas en {out_root}")
        return
    df = pd.DataFrame(rows).sort_values(["arch", "params", "lr", "batch_size"])
    df.to_csv(out_root / "resumen.csv", index=False)
    print(df.drop(columns=["run", "model_kwargs"]).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, title in zip(axes, ["acc", "acc_snr_le_m10"], ["Accuracy global (test)", "Accuracy SNR <= -10 dB (test)"]):
        for arch, g in df.groupby("arch"):
            best = g.groupby("params")[col].max().reset_index().sort_values("params")  # mejor config por tamaño
            ax.plot(best["params"], best[col] * 100, marker="o", label=arch)
        ax.set_xscale("log"); ax.set_xlabel("Parámetros"); ax.set_ylabel("Accuracy (%)"); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Mejor configuración (lr/batch/etc.) por tamaño de modelo")
    fig.tight_layout()
    fig.savefig(out_root / "accuracy_vs_parametros.png", dpi=150)
    print(f"\nEscrito: {out_root / 'resumen.csv'} y {out_root / 'accuracy_vs_parametros.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", nargs="+", choices=list(ARCHS), default=list(ARCHS))
    p.add_argument("--params", nargs="+", default=["base"], help='metas de parametros (ints) y/o "base"')
    p.add_argument("--lrs", nargs="+", default=["auto"], help='learning rates, o "auto" = default por arquitectura')
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[256])
    p.add_argument("--augment", nargs="+", type=int, choices=[0, 1], default=[1])
    p.add_argument("--curriculum", nargs="+", type=int, choices=[0, 1], default=[0])
    p.add_argument("--scheduler", nargs="+", type=int, choices=[0, 1], default=[0],
                   help="ReduceLROnPlateau (factor 0.8, patience 10), como cnn_preproc")
    p.add_argument("--repr", nargs="+", choices=["iq", "ap"], default=["iq"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=50, help="tope de epocas (con curriculum se usa max(esto, 80))")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train-subset", type=int, default=None, help="solo para pruebas rapidas")
    p.add_argument("--val-subset", type=int, default=None, help="solo para pruebas rapidas")
    p.add_argument("--out", type=Path, default=ROOT / "Resultados" / "sweep")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resumen", action="store_true", help="arma la tabla y el grafico de lo ya corrido y sale")
    args = p.parse_args()

    if args.resumen:
        resumen(args.out)
        return

    runs = build_runs(args)
    pendientes = [r for r in runs if not (args.out / run_id(r) / "metrics.json").exists()]
    print(f"Plan: {len(runs)} corridas ({len(runs) - len(pendientes)} ya hechas, {len(pendientes)} pendientes) -> {args.out}\n")
    for r in runs:
        estado = "ya hecha" if r not in pendientes else "pendiente"
        meta = "" if r["target"] == "base" else f" meta {int(r['target']):,}"
        aviso = ""
        if r["target"] != "base" and abs(r["params"] - int(r["target"])) / int(r["target"]) > 0.15:
            aviso = "  <-- AVISO: no se pudo acercar a la meta"
        print(f"  [{estado:9s}] {run_id(r)}  ({r['params']:,} params{meta}){aviso}")
    if args.dry_run:
        return
    if args.max_runs:
        pendientes = pendientes[: args.max_runs]

    X, y, snrs, _ = load_radioml_digital(dataset_folder=str(MI / "rfml"), seed=SEED_SPLIT)
    split = split_dataset(X, y, snrs, seed=SEED_SPLIT)
    for i, cfg in enumerate(pendientes, 1):
        rid = run_id(cfg)
        print(f"\n===== [{i}/{len(pendientes)}] {rid} =====", flush=True)
        m = run_one(cfg, split, args.out / rid)
        print(f"-> acc={m['accuracy_global']:.4f} | acc(SNR<=-10)={m['accuracy_snr_menor_igual_-10']:.4f} | "
              f"{m['epocas_entrenadas']} epocas | {m['segundos_entrenamiento'] / 60:.1f} min", flush=True)
    print("\nListo. Para ver la tabla: python Resultados/sweep.py --resumen")


if __name__ == "__main__":
    main()
