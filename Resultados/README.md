# Resultados

Carpeta de resultados de los experimentos. Hay tres formas de guardar, según de dónde sale la corrida:

| Qué | Dónde queda | Quién lo genera | Para qué |
|---|---|---|---|
| **Etapa** (foto comparable de un conjunto de modelos) | `Resultados/<etapa>/` | `snapshot.py` o `sweep.py` (ver más abajo) | Congelar y comparar un conjunto de modelos con una receta fija |
| **Corrida del barrido** | `Resultados/sweep/<id>/` | `sweep.py` | Probar combinaciones de arquitectura, parámetros, lr, batch, etc. |
| **Corrida de notebook** | `Resultados/sweep/<id>_nb<fecha-hora>/` | `celda_guardar_corrida.py` (pegada al final del notebook) | Guardar una corrida hecha a mano en el notebook, con el mismo formato que el barrido |

## Diferencia entre `sweep.py` y `snapshot.py`

- **`snapshot.py` congela lo que ya existe.** Toma los `.pt`, `history.pkl`, gráficos y notebooks de las carpetas `Modelos iniciales/` y `Modelos_Bajo_SNR/`, los copia a `Resultados/<etapa>/` y calcula sus métricas. No entrena nada. Sirve para tener una "foto" de los modelos que corriste a mano en Jupyter.
- **`sweep.py` entrena cosas nuevas.** Recorre una grilla de configuraciones (arquitectura, cantidad de parámetros, lr, batch, augmentation, curriculum, scheduler, fuente de datos), entrena cada una y guarda el resultado en `sweep/`. Es independiente de los notebooks: no toca sus carpetas ni sus archivos.

En corto: `snapshot` = copiar y medir modelos ya entrenados por los notebooks; `sweep` = entrenar y medir configuraciones nuevas.

## Etapas (`<etapa>/`)

Hay dos orígenes posibles para una etapa, y no se mezclan:

- **Basada en `snapshot.py`** (la original): copia `.pt`, `history.pkl`, gráficos y notebook de cada modelo.
  ```
  <etapa>/
    metricas.json                 # resumen de todos los modelos de la etapa
    Modelos iniciales/<modelo>/   # .pt, history.pkl, .png, notebook
    Modelos_Bajo_SNR/<modelo>/    # idem
  ```
  Crear una: `python Resultados/snapshot.py <nombre> --grupos "Modelos iniciales"`. Aborta si la etapa ya existe (`--sobrescribir` para forzarlo, no conviene). Los modelos nuevos hay que registrarlos en `SPECS` de `snapshot.py`, o aparecen como `SIN SPEC`.
  Etapas de este tipo: `etapa_0_original/`, `etapa_1_augmentation_curriculum/` (ambas quedaron como respaldo de un camino anterior; no son la referencia actual).

- **Basada en `sweep.py`** (la que se usa ahora para comparar etapas): no copia `.pt` ni notebooks; `metricas.json` se arma a mano a partir de las corridas de `sweep/` correspondientes (ver el campo `fuente` de cada modelo dentro de `metricas.json`). Cada una tiene su propio `README.md` con el comando exacto, la tabla de resultados y el análisis.
  ```
  <etapa>/
    metricas.json   # {"etapa": ..., "fuente": "Resultados/sweep (sweep.py)...", "modelos": {...}}
    README.md       # comando, tabla de resultados, comparación con la etapa anterior
  ```
  Etapas de este tipo, en orden (todas con batch 256, lr 1e-3, config `base` de cada arquitectura,
  test siempre RadioML real):

  | Etapa | Receta | CNN | LSTM | GRU | TCN | ULCNN |
  |---|---|---|---|---|---|---|
  | `etapa_0_baseline/` | sin augmentation, sin scheduler, sin curriculum | 0.490 | **0.179 (colapsa)** | 0.603 | 0.625 | 0.584 |
  | `etapa_1_augmentation_scheduler/` | + augmentation + scheduler | 0.521 | 0.675 | 0.677 | 0.645 | 0.647 |
  | `etapa_2_curriculum/` | + curriculum por SNR (datos reales) | 0.484 | 0.673 | 0.600 | **0.213 (colapsa)** | 0.600 |
  | `etapa_3_sintetico_curriculum/` | igual que la 2, pero train/val sintéticos | 0.458 | 0.569 | 0.560 | 0.489 | 0.556 |

  (accuracy global en el test de RadioML real; el detalle y la accuracy a SNR ≤ -10 dB están en
  el `README.md` de cada etapa)

  - **`etapa_0_baseline/`**: receta original, sin agregados. La LSTM colapsa (train loss clavado en `ln(8)`, nunca aprende) — el mismo problema que documentaron los notebooks con lr 1e-3 sin normalizar el lr.
  - **`etapa_1_augmentation_scheduler/`**: + augmentation (rotación I/Q, train 4x) + `ReduceLROnPlateau`. Mejora a las cinco arquitecturas; a la LSTM la saca del colapso.
  - **`etapa_2_curriculum/`**: + curriculum por SNR (entrenar de SNR alto a bajo). Con RadioML real, **empeora a las cinco y colapsa al TCN** (0.645 → 0.213): se sobreajusta a cada grupo de SNR y no generaliza al sumar el siguiente.
  - **`etapa_3_sintetico_curriculum/`**: misma receta que la etapa 2, pero train/val se generan on-the-fly con `Datos_Sinteticos/generador.py` (test sigue siendo RadioML real). El TCN **no** colapsa (0.489), lo que sugiere que el colapso de la etapa 2 tiene que ver con algo de RadioML real que el generador no modela (fading, offset de reloj). El resto rinde un poco peor que con datos reales (el generador no tiene esos efectos de canal).

  Falta la variante "datos sintéticos sin curriculum" (`--data-source synthetic --curriculum 0`),
  que aislaría si la diferencia de la etapa 3 viene del curriculum o de la fuente de datos.

  Para comparar accuracy entre estas cuatro, usar `Resultados/comparar_etapas.ipynb` (lee los `metricas.json`, no entrena nada).

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
- **`_sch`**: se agregó si el `ReduceLROnPlateau` estaba activo (`lr_scheduler`).
- **`_syn`**: la corrida entrenó con datos sintéticos (`--data-source synthetic`); el test sigue siendo RadioML real. Sin este sufijo, la fuente es RadioML real.
- **`_nb<fecha-hora>`** al final: corrida guardada desde un notebook (`celda_guardar_corrida.py`), no desde `sweep.py`. Lleva la fecha y hora para que dos corridas con la misma configuración no se pisen.

Si una corrida tiene la misma configuración que otra (mismo `run_id`, sin contar `_nb<fecha-hora>`), la carpeta es la misma y el barrido la salta (no la vuelve a entrenar).

### Corridas actuales

Hay más de 40 carpetas en `sweep/`, de distintos barridos (learning rate, capacidad, datos sintéticos, curriculum, algunas fallidas o duplicadas de pruebas anteriores). Para verlas todas:

```bash
python Resultados/sweep.py --resumen
```

Eso junta todas las corridas completas (con `metrics.json`) en `sweep/resumen.csv` y el gráfico `sweep/accuracy_vs_parametros.png`. Para comparar un subconjunto puntual (por ejemplo, los dos learning rate de un barrido), usar `Resultados/comparar_barrido.ipynb`, que filtra por receta en vez de mezclar todo.

Notas:

- `segundos_entrenamiento` vale 0 en las corridas guardadas desde un notebook (`_nb...`): el notebook no mide el tiempo. Las corridas de `sweep.py` sí lo miden.
- Corridas con receta distinta (augmentation, curriculum, scheduler, batch, fuente de datos) no son comparables entre sí directamente — hay que fijarse en los flags del nombre o en `config.json` antes de comparar dos corridas.

## Resumen y gráficos

`python Resultados/sweep.py --resumen` recorre `sweep/*/metrics.json` y genera:

- `sweep/resumen.csv`: una fila por corrida con arquitectura, parámetros, lr, batch, fuente de datos (`data_source`), augmentation, curriculum, accuracy global, a ≤ -10 dB, a > 0 dB, épocas y minutos.
- `sweep/accuracy_vs_parametros.png`: accuracy vs cantidad de parámetros.

Las corridas incompletas (sin `metrics.json`) se ignoran.

## Notebooks de comparación

- **`Resultados/comparar_barrido.ipynb`**: compara corridas de `sweep/` dentro de una misma receta (por ejemplo, los dos learning rate de un barrido). Filtra por `curriculum`/`batch`/`augment`/`scheduler` para no mezclar recetas distintas.
- **`Resultados/comparar_etapas.ipynb`**: compara accuracy entre las etapas basadas en `sweep.py` (`etapa_0_baseline` a `etapa_3_sintetico_curriculum`), una al lado de la otra, por arquitectura.

Ninguno de los dos entrena ni modifica nada; solo leen `metrics.json`/`metricas.json`.
