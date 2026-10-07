# Resultados

Carpeta de resultados de los experimentos. Hay tres formas de guardar, según de dónde sale la corrida:

| Qué | Dónde queda | Quién lo genera | Para qué |
|---|---|---|---|
| **Etapa** (foto de un conjunto de modelos) | `Resultados/<etapa>/` | `snapshot.py` | Congelar el estado de todos los modelos de los notebooks en un momento dado |
| **Corrida del barrido** | `Resultados/sweep/<id>/` | `sweep.py` | Probar combinaciones de arquitectura, parámetros, lr, batch, etc. |
| **Corrida de notebook** | `Resultados/sweep/<id>_nb<fecha-hora>/` | `celda_guardar_corrida.py` (pegada al final del notebook) | Guardar una corrida hecha a mano en el notebook, con el mismo formato que el barrido |

## Diferencia entre `sweep.py` y `snapshot.py`

- **`snapshot.py` congela lo que ya existe.** Toma los `.pt`, `history.pkl`, gráficos y notebooks de las carpetas `Modelos iniciales/` y `Modelos_Bajo_SNR/`, los copia a `Resultados/<etapa>/` y calcula sus métricas. No entrena nada. Sirve para tener una "foto" de cada etapa del proyecto (antes de augmentation, después, etc.) sin que se pisen entre sí.
- **`sweep.py` entrena cosas nuevas.** Recorre una grilla de configuraciones (arquitectura, cantidad de parámetros, lr, batch, augmentation, curriculum, scheduler), entrena cada una y guarda el resultado en `sweep/`. Es independiente de los notebooks: no toca sus carpetas ni sus archivos.

En corto: `snapshot` = copiar y medir modelos ya entrenados por los notebooks; `sweep` = entrenar y medir configuraciones nuevas.

## Etapas (`<etapa>/`)

Cada etapa tiene:

```
<etapa>/
  metricas.json                 # resumen de todos los modelos de la etapa
  Modelos iniciales/<modelo>/   # .pt, history.pkl, .png, notebook
  Modelos_Bajo_SNR/<modelo>/    # idem
```

`metricas.json` tiene `etapa` y `modelos`. Por cada modelo: accuracy global, accuracy por SNR, accuracy a SNR ≤ -10 dB y cantidad de parámetros. Se calcula evaluando cada modelo en el test set.

Etapas existentes:

- **`etapa_0_baseline/`**: los 13 modelos antes de augmentation y curriculum. El notebook de `Modelos iniciales` se toma de git (commit `b9f488c`).
- **`tcn_curriculum_fallido/`**: un `tcn.pt` y su `history.pkl` de una corrida con curriculum que salió mal. Se conserva como referencia; no es una etapa completa.

Crear una etapa nueva:

```bash
python Resultados/snapshot.py etapa_1_augmentation_curriculum --grupos "Modelos iniciales"
```

El script aborta si la etapa ya existe (no pisa nada). Para forzarlo está `--sobrescribir`, que no conviene usar. Los modelos nuevos hay que registrarlos en `SPECS` de `snapshot.py`; si no, aparecen como `SIN SPEC`.

## Corridas (`sweep/<id>/`)

Cada corrida es una carpeta con:

```
sweep/<id>/
  config.json     # configuración completa: arquitectura, model_kwargs, lr, batch, augmentation, curriculum, scheduler, seed, fecha
  history.json    # curvas por época (pérdida, accuracy, etc.)
  metrics.json    # resultados finales; su presencia indica que la corrida terminó
  model.pt        # pesos entrenados
```

### Cómo leer el nombre `<id>`

```
gru_iq_p40428_lr0.001_bs64_aug1_cur1_seed42
│   │   │       │      │      │      │
│   │   │       │      │      │      └─ semilla
│   │   │       │      │      └─ curriculum por SNR (1 = activo, 0 = no)
│   │   │       │      └─ augmentation de rotación I/Q (1 = activo, 0 = no)
│   │   │       └─ tamaño de batch
│   │   └─ learning rate
│   └─ cantidad de parámetros del modelo
└─ arquitectura
```

- **arquitectura**: `cnn`, `lstm`, `gru`, `tcn`, `ulcnn`.
- **repr**: `iq` o `ap` (amplitud/fase), o el nombre de la representación si es de bajo SNR. Va justo después de la arquitectura.
- **`p<params>`**: parámetros reales, no el objetivo del barrido.
- **`_sch`** al final: se agregó si el `ReduceLROnPlateau` estaba activo (`lr_scheduler`).
- **`_nb<fecha-hora>`** al final: corrida guardada desde un notebook. Lleva la fecha y hora para que dos corridas con la misma configuración no se pisen.

Si una corrida tiene la misma configuración que otra, la carpeta es la misma y el barrido la salta (no la vuelve a entrenar).

### Corridas actuales

| Id (resumido) | Params | Global | ≤ -10 dB | > 0 dB | Nota |
|---|---|---|---|---|---|
| gru_iq, bs64, cur1 | 40.428 | 0.665 | 0.155 | 0.973 | |
| tcn_iq, cur0 | 75.592 | 0.636 | 0.154 | 0.953 | sin curriculum |
| tcn_iq_ap_iq | 79.616 | 0.649 | 0.152 | 0.976 | nombre engañoso: es un TCN, no la fusión I/Q+AP |
| tcn_ap | 100.552 | 0.647 | 0.146 | 0.971 | |
| tcn_iq, cur1 | 75.596 | 0.619 | 0.134 | 0.945 | |
| ulcnn, cur0 (dos corridas) | 9.364 | 0.633 | 0.171 | 0.933 | duplicada en otra fecha |
| lstm, cur1 (dos corridas) | 208.588 | 0.608 | 0.154 | 0.867 | duplicada, mismos números |
| tcn_iq, cur1 (dos corridas) | 75.592 | 0.254 / 0.179 | 0.137 / 0.127 | 0.361 / 0.190 | **fallidas** |

Notas:

- `segundos_entrenamiento` vale 0 en todas las corridas: el tiempo no se está midiendo bien.
- Las corridas con `cur1` no son comparables directamente con las `cur0`, porque cambia la receta de entrenamiento.

## Resumen y gráficos

`python Resultados/sweep.py --resumen` recorre `sweep/*/metrics.json` y genera:

- `sweep/resumen.csv`: una fila por corrida con arquitectura, parámetros, lr, batch, augmentation, curriculum, accuracy global, a ≤ -10 dB, a > 0 dB, épocas y minutos.
- `sweep/accuracy_vs_parametros.png`: accuracy vs cantidad de parámetros.

Las corridas incompletas (sin `metrics.json`) se ignoran.
