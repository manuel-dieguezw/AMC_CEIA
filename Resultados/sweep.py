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

Parametros (todos los que aceptan varios valores -- nargs="+" -- generan el producto cartesiano
de corridas; --dry-run para ver el plan antes de entrenar):

    --archs                   que arquitecturas correr (ver ARCHS: cnn, lstm, gru, tcn, ulcnn,
                               tcn_iq_ap). cnn_ap/tcn_ap NO son claves separadas: son cnn/tcn con
                               --repr ap. Default: todas.
    --params                  metas de parametros a buscar por arquitectura: enteros (ej. 10000)
                               y/o "base" (la config actual del notebook de esa arquitectura).
                               Default: ["base"].
    --lrs                     learning rates a probar, o "auto" = el default de cada arquitectura
                               (ver ARCHS[arch]["lr"]). Default: ["auto"].
    --batch-sizes              tamaños de batch. Default: [256].
    --augment                 0/1, rotar la constelacion I/Q x4 en train (augment_rotate_iq). Se
                               ignora (queda en 0) si --data-source es "synthetic". Default: [1].
    --curriculum               0/1, activar SNRCurriculum (entrenar de SNR alto a bajo). Con
                               curriculum, epochs pasa a ser max(--epochs, 80). Default: [0].
    --curriculum-mode         solo si --curriculum 1: "acc" (original, val accuracy sobre TODOS
                               los SNR decide el corte de cada etapa), "stage_val" (el corte usa
                               val accuracy solo del rango de SNR ya visto en esa etapa; el val
                               completo se usa igual para el checkpoint final) o "fixed" (N epocas
                               fijas por grupo, sin mirar accuracy -- ver --curriculum-epochs-per-stage).
    --curriculum-epochs-per-stage  cuantas epocas fijas por grupo de SNR cuando
                               --curriculum-mode es "fixed". Default: 10.
    --scheduler                0/1, agregar ReduceLROnPlateau sobre val_loss (factor 0.8,
                               patience 10, min_lr 1e-7) -- receta de cnn_preproc. Default: [0].
    --repr                    representacion de entrada: "iq" o "ap" (to_amplitude_phase). OJO
                               con tcn_iq_ap: su forward espera I/Q crudo (calcula A/phi adentro),
                               asi que usarlo con --repr ap no tiene sentido. Default: ["iq"].
    --data-source             "radioml" (dataset real, split fijo) o "synthetic" (train/val
                               generados on-the-fly por Datos_Sinteticos/generador.py, un lote
                               nuevo por epoca; el test SIEMPRE es RadioML real para que sea
                               comparable). Default: ["radioml"].
    --synthetic-train-samples  tamaño del lote sintetico de train por epoca (solo si data-source
                               synthetic). Default: 384000 (= RadioML train aumentado 4x).
    --synthetic-val-samples   tamaño del lote sintetico de val, fijo para toda la corrida (para
                               que early stopping/scheduler comparen entre epocas). Default: 32000.
    --synthetic-escala        amplitud de la señal sintetica (std). Default: 0.0075 (~ std de
                               RadioML real, 0.006).
    --seeds                   semillas (modelo + shuffle); el split train/val/test de RadioML
                               usa siempre SEED_SPLIT=42, no esta. Default: [42].
    --epochs                  tope de epocas (early stopping corta antes salvo en curriculum
                               "fixed"). Con curriculum se usa max(--epochs, 80). Default: 50.
    --patience                paciencia del EarlyStopping sobre val_loss (se ignora con
                               curriculum). Default: 10.
    --train-subset            recorta el train a N muestras al azar -- solo para pruebas rapidas,
                               no para corridas que se vayan a comparar. Default: sin recortar.
    --val-subset               idem para val. Default: sin recortar.
    --device                  "cpu" (default), "cuda", o "dml" (DirectML -- requiere correr este
                               script con el python de un venv con torch_directml instalado, ver
                               Modelos iniciales/_smoke/smoke_directml.py). No es parte del
                               run_id: no afecta que corridas se saltean como "ya hechas".
    --out                     carpeta donde se guardan/leen las corridas. Default:
                               Resultados/sweep/.
    --max-runs                 corre como maximo N de las corridas pendientes (para probar antes
                               de lanzar un barrido grande). Default: sin limite.
    --dry-run                 solo imprime el plan (corridas, parametros reales, cuales ya estan
                               hechas) y sale, sin entrenar nada.
    --resumen                 arma Resultados/sweep/resumen.csv + accuracy_vs_parametros.png con
                               TODO lo que haya en --out y sale, sin entrenar nada.

Uso (con el python del venv del proyecto):

    # ver el plan sin entrenar nada (cantidad real de parametros de cada corrida y cuales ya existen)
    python Resultados/sweep.py --archs cnn tcn ulcnn --params base 10000 50000 --dry-run

    # correr
    python Resultados/sweep.py --archs cnn tcn gru lstm ulcnn --params 10000 50000 \\
        --lrs 1e-3 3e-4 --batch-sizes 128 256 --augment 1 --curriculum 0 1

    # tabla + grafico accuracy vs parametros con todo lo que haya en Resultados/sweep/
    python Resultados/sweep.py --resumen

    # con datos sinteticos (Datos_Sinteticos/generador.py): train/val generados on-the-fly,
    # test SIEMPRE es RadioML real, para que la comparacion con las corridas "radioml" sea valida
    python Resultados/sweep.py --archs cnn tcn ulcnn --params base --data-source synthetic

    # con GPU via DirectML (Windows, AMD/Intel): correr con el python de un venv aparte que
    # tenga torch_directml instalado (no soporta Python 3.13, usar 3.12 -- ver
    # Modelos iniciales/_smoke/smoke_directml.py para un test aislado de velocidad)
    amc_env_dml/Scripts/python.exe Resultados/sweep.py --archs tcn --params base --device dml
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
sys.path.insert(0, str(ROOT / "Datos_Sinteticos"))

from common.data import load_radioml_digital, split_dataset, make_loader, to_amplitude_phase, augment_rotate_iq
from common.models import CNNBaseline, LSTMClassifier, GRUClassifier, TCN, ULCNN, DualTCN
from common.train import train_model, SNRCurriculum, evaluate_by_snr, overall_accuracy
from generador import generar_lote, config as generador_config

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
    # DualTCN (I/Q + A/phi, dos ramas con cabeza comun -- ver Modelos iniciales/tcn_iq_ap/ y
    # common/models.py). Recibe SIEMPRE I/Q crudo (calcula A/phi adentro del forward), asi que
    # --repr para esta arquitectura tiene que quedar en "iq" (el default); "ap" le daria una
    # representacion ya transformada a una rama que espera I/Q y no tiene sentido.
    "tcn_iq_ap": dict(cls=DualTCN, widths=range(2, 257, 1), lr=1e-3,
                       make=lambda w: dict(channels=w, dilations=(1, 2, 4, 8), dropout=0.2),
                       base=dict(channels=40, dilations=(1, 2, 4, 8), dropout=0.2)),
}

_width_cache = {}


def resolve_device(name):
    """"cpu" (default), "cuda", o "dml" (DirectML, para GPUs AMD/Intel en Windows).

    "dml" requiere correr este script con el python de un venv que tenga `torch_directml`
    instalado (p. ej. amc_env_dml, Python 3.12 -- torch_directml no soporta 3.13). No mezclar
    "cpu"/"cuda" y "dml" en el mismo proceso: el motor de autograd de PyTorch rompe si ve un
    device nuevo despues de haber corrido backward() en otro (bug conocido de torch-directml).
    Acá no es un problema porque todas las corridas de una misma invocacion usan el mismo device.
    """
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda pedido pero CUDA no esta disponible en este entorno")
        return "cuda"
    if name == "dml":
        try:
            import torch_directml
        except ImportError:
            raise SystemExit(
                "--device dml pedido pero torch_directml no esta instalado en este entorno. "
                "Correr este script con el python de amc_env_dml (ver Modelos iniciales/_smoke/)."
            )
        return torch_directml.device()
    raise SystemExit(f"device desconocido: {name}")


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
    fuente = "_syn" if cfg["data_source"] == "synthetic" else ""
    # sufijo de modo de curriculum: solo cuando no es el modo original ("acc") para no romper IDs existentes
    cur_mode = ""
    if cfg.get("curriculum"):
        mode = cfg.get("curriculum_mode", "acc")
        if mode == "stage_val":
            cur_mode = "_cvsv"
        elif mode == "fixed":
            cur_mode = "_cvfix"
    return (f"{cfg['arch']}_{cfg['repr']}_p{cfg['params']}_lr{lr}_bs{cfg['batch_size']}"
            f"_aug{int(cfg['augment'])}_cur{int(cfg['curriculum'])}{cur_mode}{'_sch' if cfg.get('lr_scheduler') else ''}"
            f"{fuente}_seed{cfg['seed']}")


def build_runs(args):
    runs = []
    for arch, target, repr_, bs, ds, cur, sch, seed in itertools.product(
            args.archs, args.params, args.repr, args.batch_sizes, args.data_source, args.curriculum, args.scheduler, args.seeds):
        kw, n = kwargs_for_target(arch, target)
        lrs = [ARCHS[arch]["lr"]] if args.lrs == ["auto"] else [float(x) for x in args.lrs]
        # augment_rotate_iq (rotar la constelación 4x) es una augmentation de datos *reales* ya fijos;
        # con datos sinteticos no aporta nada, porque el generador ya da un lote nuevo cada vez.
        augs = args.augment if ds == "radioml" else [0]
        for lr, aug, cur_mode in itertools.product(lrs, augs, args.curriculum_mode):
            epochs = max(args.epochs, 80) if cur else args.epochs
            runs.append(dict(arch=arch, repr=repr_, target=target, params=n, model_kwargs=jsonable(kw), lr=lr,
                             batch_size=bs, epochs=epochs, patience=args.patience, augment=bool(aug),
                             curriculum=bool(cur), curriculum_mode=cur_mode,
                             curriculum_epochs_per_stage=args.curriculum_epochs_per_stage,
                             lr_scheduler=bool(sch), seed=seed, data_source=ds,
                             synthetic_train_samples=args.synthetic_train_samples,
                             synthetic_val_samples=args.synthetic_val_samples,
                             synthetic_escala=args.synthetic_escala,
                             train_subset=args.train_subset, val_subset=args.val_subset))
    unicas = {}
    for r in runs:
        unicas.setdefault(run_id(r), r)
    return list(unicas.values())


def prepare_data(cfg, split):
    (Xtr, ytr, str_), (Xva, yva, snr_va), (Xte, yte, ste) = split
    if cfg["data_source"] == "synthetic":
        rng = np.random.default_rng(cfg["seed"])
        Xva, yva, snr_va = generar_lote(cfg["synthetic_val_samples"], rng=rng, escala=cfg["synthetic_escala"])
        if cfg["curriculum"]:
            Xtr, ytr, str_ = generar_lote(cfg["synthetic_train_samples"], rng=rng, escala=cfg["synthetic_escala"])
        else:
            Xtr, ytr, str_ = None, None, None
    elif cfg["augment"]:
        Xtr, ytr, str_ = augment_rotate_iq(Xtr, ytr, str_)
    if cfg["repr"] == "ap":
        Xva, Xte = to_amplitude_phase(Xva), to_amplitude_phase(Xte)
        if Xtr is not None:
            Xtr = to_amplitude_phase(Xtr)
    rng = np.random.default_rng(0)
    if cfg["train_subset"] and Xtr is not None:
        i = rng.choice(len(Xtr), min(cfg["train_subset"], len(Xtr)), replace=False)
        Xtr, ytr, str_ = Xtr[i], ytr[i], str_[i]
    if cfg["val_subset"]:
        i = rng.choice(len(Xva), min(cfg["val_subset"], len(Xva)), replace=False)
        Xva, yva, snr_va = Xva[i], yva[i], snr_va[i]
    return (Xtr, ytr, str_), (Xva, yva, snr_va), (Xte, yte, ste)


def run_one(cfg, split, out_dir, device="cpu"):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    (Xtr, ytr, str_), (Xva, yva, snr_va), (Xte, yte, ste) = prepare_data(cfg, split)

    model = ARCHS[cfg["arch"]]["cls"](n_classes=N_CLASSES, **{k: tuple(v) if isinstance(v, list) else v
                                                              for k, v in cfg["model_kwargs"].items()})
    val_loader = make_loader(Xva, yva, batch_size=512, shuffle=False)
    if cfg["curriculum"]:
        cur_mode = cfg.get("curriculum_mode", "acc")
        cur_kwargs = dict(batch_size=cfg["batch_size"], mode=cur_mode,
                         epochs_per_stage=cfg.get("curriculum_epochs_per_stage", 10))
        if cur_mode == "stage_val":
            cur_kwargs.update(X_val=Xva, y_val=yva, snr_val=snr_va)
        curriculum = SNRCurriculum(Xtr, ytr, str_, **cur_kwargs)
    else:
        curriculum = None

    train_loader_fn = None
    if cfg["data_source"] == "synthetic" and not cfg["curriculum"]:
        # Semilla distinta a la del val (que ya consumió cfg["seed"] en prepare_data) para no
        # repetir la misma secuencia de ruido pseudoaleatorio.
        gen_rng = np.random.default_rng(cfg["seed"] + 1_000_000)
        repr_ap = cfg["repr"] == "ap"

        def train_loader_fn():
            Xb, yb, _ = generar_lote(cfg["synthetic_train_samples"], rng=gen_rng, escala=cfg["synthetic_escala"])
            if repr_ap:
                Xb = to_amplitude_phase(Xb)
            return make_loader(Xb, yb, batch_size=cfg["batch_size"], shuffle=True)

        train_loader = None
    else:
        train_loader = make_loader(Xtr, ytr, batch_size=cfg["batch_size"], shuffle=True)

    t0 = time.time()
    model, history = train_model(model, train_loader, val_loader, epochs=cfg["epochs"], lr=cfg["lr"],
                                 patience=cfg["patience"], curriculum=curriculum, device=device,
                                 use_lr_scheduler=cfg.get("lr_scheduler", False),
                                 scheduler_factor=0.8, scheduler_patience=10, scheduler_min_lr=1e-7,
                                 train_loader_fn=train_loader_fn)
    train_seconds = time.time() - t0

    by_snr = evaluate_by_snr(model, Xte, yte, ste, device=device)
    low, pos = ste <= -10, ste > 0
    metrics = {
        "accuracy_global": overall_accuracy(model, Xte, yte, device=device),
        "accuracy_snr_menor_igual_-10": overall_accuracy(model, Xte[low], yte[low], device=device),
        "accuracy_snr_mayor_0": overall_accuracy(model, Xte[pos], yte[pos], device=device),
        "accuracy_por_snr": {str(int(k)): float(v) for k, v in by_snr.items()},
        "parametros": int(model.count_parameters()),
        "epocas_entrenadas": len(history["train_loss"]),
        "mejor_val_acc": float(max(history["val_acc"])),
        "mejor_val_loss": float(min(history["val_loss"])),
        "segundos_entrenamiento": round(train_seconds, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = dict(cfg, fecha=datetime.now().isoformat(timespec="seconds"),
                   torch=torch.__version__, python=platform.python_version(), device=str(device))
    if cfg["data_source"] == "synthetic":
        cfg_out["generador_config"] = generador_config(semilla=cfg["seed"], escala=cfg["synthetic_escala"])
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2, ensure_ascii=False), encoding="utf-8")
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
                         batch_size=cfg["batch_size"], data_source=cfg.get("data_source", "radioml"),
                         augment=cfg["augment"], curriculum=cfg["curriculum"],
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
    p.add_argument("--curriculum-mode", nargs="+", choices=["acc", "stage_val", "fixed"], default=["acc"],
                   help='"acc": original (val completo); "stage_val": early stopping por rango de SNR visible; '
                        '"fixed": N epocas fijas por grupo sin mirar accuracy')
    p.add_argument("--curriculum-epochs-per-stage", type=int, default=10,
                   help="epocas fijas por grupo cuando --curriculum-mode fixed (default: 10)")
    p.add_argument("--scheduler", nargs="+", type=int, choices=[0, 1], default=[0],
                   help="ReduceLROnPlateau (factor 0.8, patience 10), como cnn_preproc")
    p.add_argument("--repr", nargs="+", choices=["iq", "ap"], default=["iq"])
    p.add_argument("--data-source", nargs="+", choices=["radioml", "synthetic"], default=["radioml"],
                   help='"synthetic" entrena con Datos_Sinteticos/generador.py (test SIEMPRE es RadioML real)')
    p.add_argument("--synthetic-train-samples", type=int, default=384000,
                   help="tamaño del train sintetico por epoca (default: igual al de RadioML aumentado 4x)")
    p.add_argument("--synthetic-val-samples", type=int, default=32000)
    p.add_argument("--synthetic-escala", type=float, default=0.0075,
                   help="amplitud de la señal sintetica; ~0.0075 imita la de RadioML (std ~0.006)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=50, help="tope de epocas (con curriculum se usa max(esto, 80))")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train-subset", type=int, default=None, help="solo para pruebas rapidas")
    p.add_argument("--val-subset", type=int, default=None, help="solo para pruebas rapidas")
    p.add_argument("--device", choices=["cpu", "cuda", "dml"], default="cpu",
                   help='"dml" = DirectML (GPU AMD/Intel en Windows); requiere correr este script '
                        "con el python de un venv con torch_directml instalado (no amc_env)")
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

    device = resolve_device(args.device)
    print(f"Device: {args.device} ({device})")

    X, y, snrs, _ = load_radioml_digital(dataset_folder=str(MI / "rfml"), seed=SEED_SPLIT)
    split = split_dataset(X, y, snrs, seed=SEED_SPLIT)
    for i, cfg in enumerate(pendientes, 1):
        rid = run_id(cfg)
        print(f"\n===== [{i}/{len(pendientes)}] {rid} =====", flush=True)
        m = run_one(cfg, split, args.out / rid, device=device)
        print(f"-> acc={m['accuracy_global']:.4f} | acc(SNR<=-10)={m['accuracy_snr_menor_igual_-10']:.4f} | "
              f"{m['epocas_entrenadas']} epocas | {m['segundos_entrenamiento'] / 60:.1f} min", flush=True)
    print("\nListo. Para ver la tabla: python Resultados/sweep.py --resumen")


if __name__ == "__main__":
    main()
