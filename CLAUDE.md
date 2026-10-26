# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AMC_CEIA — proyecto final de la Carrera de Especialización en Inteligencia Artificial (CEIA - FIUBA). Clasificación automática de modulaciones (AMC) a partir de las componentes I/Q de señales de radio, con el dataset **RadioML 2016.10a** (subset de 8 modulaciones digitales, SNR de -20 a 18 dB). Se comparan arquitecturas de deep learning (CNN, LSTM, GRU, TCN, ULCNN), evaluando accuracy en función del SNR.

**Objetivo real**: no es solo maximizar accuracy — es encontrar el modelo **más liviano** que cumpla el objetivo, pensando en despliegue en tiempo real / embebidos. Priorizar también parámetros, FLOPs, latencia y tamaño, no solo accuracy vs SNR.

El trabajo vive en notebooks Jupyter; no hay app ni paquete instalable (aparte del venv `amc_env/` y la librería externa `rfml`). El usuario escribe en español rioplatense; responder en español.

## Reglas de trabajo con este usuario (leer primero)

- **Tocar solo lo que el usuario nombra.** No regenerar notebooks enteros ni hacer pasadas amplias "por consistencia": una regeneración temprana borró los outputs guardados de `gru.ipynb` (irrecuperable, nunca estuvo en git). Editar notebooks solo con `NotebookEdit` (celda puntual, preserva los outputs del resto).
- **Si el usuario está corriendo notebooks, no editarlos**: darle la celda para que la pegue él (así se hizo con `Resultados/celda_guardar_corrida.py`). Antes de editar un notebook hay que leerlo con `Read` (la herramienta lo exige y los ids de celda cambian: Jupyter los regenera al guardar, p. ej. `gru.ipynb` tiene ids tipo `e8424eb7`, `lstm.ipynb` usa `cell-N`).
- **Jupyter cachea los módulos**: si se cambia algo en `Modelos iniciales/common/` mientras hay un kernel abierto, el notebook sigue usando la versión vieja (error típico: `unexpected keyword argument`). Hay que reiniciar el kernel.
- **No lanzar entrenamientos largos** (CPU, dataset aumentado 4x: cada época tarda minutos). Validar con subconjuntos chicos, pocas épocas y una carpeta temporal `Modelos iniciales/_smoke/` (hermana de `rfml/` para que `../rfml` resuelva) que se borra al terminar. Cuando el código de una celda se prueba, ejecutar el código real del notebook (leer el `.ipynb` con `json`, `exec` de las celdas), no una copia.
- **No commitear ni pushear** salvo pedido explícito. `amc_env/` y `*.pt` están en `.gitignore` y nunca quedaron trackeados, así que no aparecen en `git status`. `CLAUDE.md` sí está en `.gitignore` pero quedó trackeado de antes de agregarlo (sigue apareciendo como modificado); no destrackearlo sin pedido explícito.
- Los `.png`/`history.pkl`/`.pt` de cada notebook se **sobrescriben** al reentrenar. Antes de reentrenar una etapa, hacer snapshot (ver `Resultados/`).
- Shell en Windows/bash: heredocs con comillas complicadas fallan; si un comando es largo, escribir el script a un archivo y ejecutarlo. Los scripts que cargan datos deben correr con cwd = carpeta hermana de `rfml/` o pasar rutas absolutas (`load_radioml_digital(dataset_folder=...)`).

## Setup y comandos

```bash
.\amc_env\Scripts\activate        # Windows (venv ya existe en amc_env/)
pip install -r requirements.txt   # incluye PyWavelets (requirements.txt regenerado en UTF-8 con pip freeze)
```

No hay build, lint ni tests: "run" = ejecutar celdas de notebook o los scripts de `Resultados/`. Python del venv: `D:/CEIA/Proyecto_AMC/amc_env/Scripts/python.exe`.

### GPU / DirectML (AMD RX 6700, opcional)

No hay CUDA (GPU AMD). Para acelerar con DirectML hay un venv aparte, `amc_env_dml/` (Python 3.12 —
`torch-directml` no soporta 3.13; creado con `uv venv amc_env_dml --python 3.12`), con
`torch==2.4.1+cpu` + `torch_directml` (fijo, no actualizar junto al resto: instalar el resto de
`requirements.txt` ahí pisaría esa versión y rompería DirectML). Le falta instalar paquete por
paquete lo que haga falta de `requirements.txt` sin tocar `torch`/`torchvision`/`torchaudio`
(`matplotlib`, `seaborn`, `pandas`, `ipykernel`, etc., según qué se vaya a correr). Los 8 notebooks
de `Modelos iniciales/` (los 7 + `tcn_iq_ap`) y `Resultados/sweep.py` (`--device dml`) tienen
fallback automático a `torch_directml.device()` si no hay CUDA y el paquete está instalado; en
`amc_env` normal (sin `torch_directml`) el comportamiento no cambia.

**Bug conocido: `LSTMClassifier`/`GRUClassifier` (`nn.LSTM`/`nn.GRU`) no corren en DirectML** —
`RuntimeError`/`NotImplementedError` en `aten::_thnn_fused_lstm_cell` (kernel fused que solo
existe para CUDA; el fallback a CPU de DirectML no lo tiene). No hay forma de esquivarlo sin
reescribir esas clases con celdas manuales. **Correr `lstm`/`gru` siempre con `--device cpu`
(o `amc_env` normal); `cnn`/`tcn`/`ulcnn`/`cnn_preproc`/`cnn_ap`/`tcn_ap`/`tcn_iq_ap` sí
funcionan con `--device dml`.**

Además, una corrida de `cnn` con `--curriculum-mode fixed` + `--scheduler 1` en DirectML mostró
`Val Loss` muy inestable (saltos a >100 entre épocas, acc final bajo) — sin confirmar todavía si
es un problema numérico de DirectML o del modo `fixed` en sí (pendiente: repetir la misma config
con `--device cpu` y comparar).

### Datos

`radioml_X.npy` / `radioml_X_digital.npy` no están versionados (>100 MB; ver README para descargarlos) y viven en `Modelos iniciales/rfml/`, junto con los `.parquet` de metadata. Los notebooks los leen por ruta relativa `../rfml/` (desde `Modelos iniciales/<x>/`). **La señal tiene amplitud minúscula (std ≈ 0.006)**; el modelo nunca recibe el SNR como entrada (solo I/Q o A/φ) — el SNR se usa solo para evaluar y, en el curriculum, para elegir qué muestras de train entran.

## Estructura del repo

```
Modelos iniciales/   rfml/ (baseline con lib externa + datos), common/ (código compartido),
                     cnn/ lstm/ gru/ tcn/ cnn_ap/ tcn_ap/ cnn_preproc/ tcn_iq_ap/ (un notebook por modelo), comparacion/
Resultados/          etapa_0_baseline/ .. etapa_6_params_iguales/ (basadas en sweep.py, con README propio),
                     etapa_0_original/ y etapa_1_augmentation_curriculum/ (legacy, basadas en snapshot.py),
                     tcn_curriculum_fallido/ (corrida descartada), snapshot.py, sweep.py, celda_guardar_corrida.py,
                     sweep/ (corridas del barrido y de notebooks), comparar_barrido.ipynb, comparar_etapas.ipynb
Datos_Sinteticos/    generador.py (señales I/Q sintéticas, numpy/scipy), validar.py, README.md — conectado a
                     Resultados/sweep.py (--data-source synthetic) y a los 7 notebooks de Modelos iniciales/
                     (DATA_SOURCE="synthetic" en la celda de config)
```

### `Modelos iniciales/common/` (única fuente de verdad; ningún notebook redefine modelos, datos ni loop)

- `models.py` — `CNNBaseline`, `LSTMClassifier`, `GRUClassifier`, `TCN` (+`ResidualBlock`), `ULCNN` (+`ComplexConv1d`, `IQCF`, `ChannelShuffle`, `ChannelAttention`, `FMDR`). Configurables por kwargs (`channels`, `hidden_size`, `num_layers`, `dilations`, `n_cv`, `n_fmdr`...); los defaults reproducen las arquitecturas con pesos ya entrenados. `LSTMClassifier`/`GRUClassifier` tienen `input_norm` (default `False`, `BatchNorm1d` sobre I/Q antes de la recurrencia): **sin normalizar, LSTM/GRU casi no aprenden** (señal ≈ 0.005) — CNN/TCN se salvan porque tienen BatchNorm tras la primera conv. `ULCNN` tiene ~9k parámetros (vs 40k–1M del resto).
- `data.py` — `load_radioml_digital`, `split_dataset` (60/20/20 estratificado, seed 42, mezcla previa porque el dataset viene ordenado por clase), `make_loader`, `to_amplitude_phase`, `augment_rotate_iq` (rotación de constelación π/2, π, 3π/2 → train 4x; solo en train, y **antes** de pasar a A/φ).
- `train.py` — `train_model` (Adam + `EarlyStopping` sobre val loss; `use_lr_scheduler=True` agrega `ReduceLROnPlateau`; `curriculum=` activa el entrenamiento por grupos de SNR). **`curriculum` y `use_lr_scheduler` se pueden combinar** (el scheduler avanza una vez por época sobre `val_loss`, igual que sin curriculum; antes tiraba `ValueError`, se sacó esa restricción). `SNRCurriculum`, `evaluate_by_snr`, `overall_accuracy`, `compute_cm`. Sin curriculum, el camino es idéntico al original (verificado época a época).

### Curriculum por SNR (`SNRCurriculum`)

Idea del usuario/tutor: entrenar de SNR alto a bajo y cortar cuando ya no hay progreso (las muestras de SNR muy bajo, ~30% del dataset, dan gradientes poco informativos). Reglas: los 20 niveles de SNR se agrupan de a `group_size=4` (5 etapas); cada etapa **suma** el grupo siguiente (más bajo) y entrena hasta `patience=3` épocas sin mejorar la **val accuracy sobre todos los SNR** en más de `min_gain=0.005` (o `max_epochs_per_stage=15`); si una etapa no mejora la mejor accuracy, se corta y se devuelven los mejores pesos. Se usa accuracy y no loss porque al sumar SNR muy bajos la loss puede mejorar solo por calibración. Guarda de arranque: mientras la mejor accuracy no supere el azar + 0.05 no se cuenta "sin progreso" (tope `max_warmup_epochs=30`, con aviso). `history.pkl`/`history.json` guarda `curriculum`, `curriculum_stage` y `curriculum_min_snr` por época. El checkpoint se elige por val accuracy. `cnn_preproc` SÍ está conectado al curriculum ahora (flag `USE_CURRICULUM` igual que el resto, combinado con su scheduler).

**Resultado con la receta común (augmentation + scheduler, lr 1e-3, batch 256, ver `Resultados/etapa_2_curriculum/README.md`): el curriculum empeora las 5 arquitecturas y colapsa al TCN** (0.645 → 0.213 con RadioML real). Con datos sintéticos el TCN no colapsa (`Resultados/etapa_3_sintetico_curriculum/README.md`), lo que sugiere que el colapso tiene que ver con algo de RadioML real (fading/offset de reloj que el generador no modela) más que con el curriculum en sí. Conclusión: **no usar curriculum en la comparación principal**.

### Notebooks de `Modelos iniciales/` (cada uno entrena UN modelo, importa todo de `common/`)

Estructura común: imports/config (`MODEL_KWARGS`, `LR`, `SCHEDULER`, `USE_CURRICULUM`...) → datos (`AUGMENT` como flag, guarda `augment_rotate_iq` detrás de un `if`) → entrenamiento → evaluación → gráficos (curva de pérdida, accuracy vs SNR, matriz de confusión global y por SNR <0 / >0 dB). Guardan `<nombre>.pt` + `history.pkl` (con `model_kwargs`) en su carpeta, y además una corrida en `Resultados/sweep/` (celda final, lee los flags de arriba). `cnn_ap`/`tcn_ap` = mismas `CNNBaseline`/`TCN` sobre A/φ. `cnn_preproc` = `ULCNN`, con su receta original (batch 256, 200 épocas, scheduler) + los mismos flags que el resto.

**Receta común unificada (2026-10-07), con flags en la celda de configuración/datos, default `AUGMENT = True`, `SCHEDULER = True`, `USE_CURRICULUM = False`**: batch 256, lr 1e-3 para las 7 (se sacó la excepción de 3e-4 de la LSTM), 200 épocas tope, `PATIENCE = 10`. `tcn` y `tcn_ap` quedaron con la misma arquitectura (`channels=64, dilations=(1,2,4), dropout=0.4, input_norm=True`), para que la única diferencia entre ambos sea la representación. **Estos 7 notebooks tienen el código actualizado pero no están reentrenados con esta receta** (los `.pt`/`history.pkl` son de corridas anteriores, con flags o lr distintos) — las etapas comparables de verdad (0 a 3) se armaron con `sweep.py`, no con estos notebooks. `comparacion/comparacion.ipynb` no entrena: reconstruye cada modelo desde su `history.pkl` (`model_kwargs`) y lo evalúa con su representación; `MODEL_SPECS` lista los 7 modelos.

### `Resultados/` — guardar etapas y barridos sin pisar nada

Ver `Resultados/README.md` para el detalle completo (cómo leer el nombre de una corrida, qué
archivos tiene cada una, etc.). Resumen:

- **Etapas basadas en `sweep.py` (la referencia actual)**: `etapa_0_baseline/` (sin augmentation/scheduler/curriculum) → `etapa_1_augmentation_scheduler/` (+ ambos) → `etapa_2_curriculum/` (+ curriculum, criterio de corte `acc`, colapsa al TCN — ver "Correcciones del tutor") → `etapa_3_sintetico_curriculum/` (datos sintéticos + curriculum, test RadioML real; obsoleta, ver nota en "Correcciones del tutor") → `etapa_4_curriculum_fixed/` (curriculum con corte fijo por épocas, `--curriculum-mode fixed`: evita el colapso, el mejor de los tres modos de curriculum) → `etapa_5_curriculum_stage_val/` (curriculum con validación por rango de SNR visto, `--curriculum-mode stage_val`: evita el colapso pero corta demasiado temprano, peor que `fixed` en 4 de 5 arquitecturas) → `etapa_6_params_iguales/` (sin curriculum, pero `cnn`/`tcn`/`gru`/`lstm` llevadas con `--params 9364` a la capacidad de `ULCNN`: a igual tamaño, ULCNN gana claro). Cada una tiene `metricas.json` (armado a mano desde las corridas de `sweep/` citadas en `fuente`) y su propio `README.md` con el comando, la tabla y el análisis. `Resultados/comparar_etapas.ipynb` las compara entre sí. **Todas cubren solo 5 arquitecturas en I/Q** (`cnn`, `lstm`, `gru`, `tcn`, `ulcnn`): falta correr `cnn_ap`/`tcn_ap` (`--repr ap`) con la misma receta.
- **Etapas basadas en `snapshot.py` (legacy)**: `etapa_0_original/` y `etapa_1_augmentation_curriculum/`. Quedaron de respaldo; no son la referencia para comparar ahora, porque otro camino (los notebooks sin reentrenar con la receta final) las dejó desactualizadas. `etapa_0_original/` además tiene, como registro histórico ya congelado, una subcarpeta `Modelos_Bajo_SNR/` con 6 modelos de una línea de exploración (HOS, STFT, wavelet, CNN+LSTM, STFT+ULCNN, fusión) que ya no se mantiene — el código vivo de esa carpeta se borró del repo; esos resultados quedan solo como dato histórico en `etapa_0_original/metricas.json`.
- `snapshot.py <etapa> [--sobrescribir]` — copia `.pt`/`history.pkl`/notebook de las carpetas de `Modelos iniciales/` + recalcula métricas en el test set. Aborta si la etapa ya existe. **No descubre modelos nuevos solos**: cada uno se registra en `SPECS`; avisa con `SIN SPEC` si encuentra carpetas no registradas.
- `sweep.py` — barrido independiente de los notebooks; cada corrida en `Resultados/sweep/<id>/` (`config.json` completo con `data_source`, `lr_scheduler`, etc.; `metrics.json`, `history.json`, `model.pt`), nunca pisa (salta lo ya hecho), `--dry-run` muestra el plan, `--params 10000 50000 base` compara arquitecturas a **igual cantidad de parámetros**, `--scheduler 0 1` activa `ReduceLROnPlateau` (0.8, patience 10), `--curriculum 0 1` activa `SNRCurriculum` (se puede combinar con `--scheduler 1`), `--data-source radioml synthetic` (sintético: train/val on-the-fly con `Datos_Sinteticos/generador.py`, test siempre RadioML real; `--synthetic-train-samples`/`--synthetic-val-samples`/`--synthetic-escala`; `augment` se ignora con datos sintéticos), `--resumen` arma `resumen.csv` + gráfico accuracy vs parámetros. Cubre `cnn`, `lstm`, `gru`, `tcn` y `ulcnn` (`cnn_ap`/`tcn_ap` = `--repr ap`). Los `base` de `ARCHS` son constantes escritas a mano: mantenerlos alineados con los notebooks. Hay más de 40 corridas guardadas, de barridos distintos (lr, capacidad, curriculum, sintético, algunas fallidas/duplicadas de pruebas); usar `--resumen` o `comparar_barrido.ipynb` para filtrar, no asumir que todas son comparables entre sí.
- `celda_guardar_corrida.py` — plantilla genérica para notebooks nuevos (guarda la corrida en `Resultados/sweep/`, mismo formato, no pisa). En los 7 notebooks de `Modelos iniciales/` ya está adaptada e integrada: lee `LR`/`AUGMENT`/`SCHEDULER`/`USE_CURRICULUM` directamente de las celdas de arriba, no hay que editarla a mano.

## Estado y pendientes (feedback del tutor, 5 puntos)

1. **Augmentation pareja** (comparar CNN vs `cnn_preproc` sin confundir preprocesamiento con arquitectura): aplicada a los 7 notebooks de `Modelos iniciales/` como flag (`AUGMENT`); la comparación real a igual receta vive en las etapas de `sweep.py` (0 a 3, ver arriba), que ya la tienen pareja. **Los notebooks siguen sin reentrenarse con la receta final** (lr unificado, `tcn`/`tcn_ap` con la misma arquitectura).
2. **Curriculum por SNR**: implementado, combinable con el scheduler, con tres modos de corte (`acc`/`stage_val`/`fixed`, ver "Correcciones del tutor") — **los tres probados**. El original (`acc`) colapsaba al TCN y empeoraba las demás; `fixed` (etapa 4) evita el colapso y queda cerca (o mejor) que sin curriculum; `stage_val` (etapa 5) también evita el colapso pero corta demasiado temprano y queda peor que `fixed` en 4 de 5 arquitecturas. **Conclusión para el informe: no usar curriculum en la comparación principal** (ninguno de los tres modos mejora de forma clara y consistente); si se reporta uno como el "correcto", usar `fixed`.
3. **Datos sintéticos on-the-fly**: generador listo y **conectado a `Resultados/sweep.py`** (`--data-source synthetic`) y **a los 7 notebooks de `Modelos iniciales/`** (`DATA_SOURCE`/`SYNTHETIC_*` en la celda de config + `train_loader_fn` conectado a `train_model`, mismo hook que usa `sweep.py`; `curriculum` + `synthetic` combinados no está soportado, bloqueado con un `assert`). Resultados reales en `etapa_3_sintetico_curriculum/` (obsoleta, ver "Correcciones del tutor") y en `Resultados/sweep/`. No tiene canal con fading ni offset de reloj como RadioML; eso probablemente explica por qué rinde algo peor que RadioML real salvo en el TCN.
4. **Sweep de hiperparámetros/capacidad**: herramienta lista y usada para armar las etapas 0-3 y el barrido de lr (`Resultados/sweep/README.md`). **Resuelto (2026-10-26)**: barrido a igual número de parámetros (`--params 9364`, llevando `cnn`/`tcn`/`gru`/`lstm` a la capacidad de `ULCNN`) — ver `Resultados/etapa_6_params_iguales/README.md`. A esa capacidad **ULCNN gana claro** (0.647 global) sobre GRU (0.594), LSTM (0.532), TCN (0.481) y CNN (0.387, la que peor aguanta la poda — corta a las 12 épocas sin converger). No es solo la arquitectura más chica en su configuración `base`: es también la que mejor aprovecha una capacidad chica cuando se fuerza a las demás a bajar a su tamaño.
5. **I/Q + A/φ juntos** (dos ramas + cabeza común): **resuelto** — `DualTCN` (`common/models.py`, notebook `tcn_iq_ap/`) recibe I/Q crudo, calcula A/φ internamente y combina ambas ramas antes de clasificar. Entrenado y evaluado (acc global 0.649, 79.616 params — ver `Resultados/sweep/tcn_iq_ap_iq_p79616_...`), integrado a `comparacion/comparacion.ipynb`, `Resultados/snapshot.py` y `ARCHS` de `Resultados/sweep.py` (2026-10-18). Pendiente menor: todavía no tiene `SCHEDULER` en su notebook (a diferencia de los otros 7 — ver nota en el notebook).

Resultados de `etapa_0_original` (legacy, 13 modelos — 7 de `Modelos iniciales` + 6 de una línea de exploración para bajo SNR ya descontinuada, accuracy global / a SNR ≤ -10 dB, test): TCN A/φ 0.653/0.168 · ULCNN 0.616/0.161 (9.364 params) · CNN+LSTM 0.595/0.164 · TCN 0.576/0.144 · CNN A/φ 0.521/0.141 · CNN 0.513/0.144 · HOS 0.442/0.128 · GRU 0.411/0.141 · STFT 0.231 · fusión 0.196 · LSTM 0.179 · wavelet 0.177 · STFT+ULCNN2D 0.123. **A SNR ≤ -10 dB todos están en el azar (≈0.125)**. Estos números son de la receta original (sin augmentation/scheduler/curriculum); para los 5 principales (`cnn`, `lstm`, `gru`, `tcn`, `ulcnn`) la referencia actual son las etapas 0-3 basadas en `sweep.py` (`Resultados/etapa_*/README.md`), no estos valores.

## Correcciones del tutor (2026-10-17)

Dos observaciones sobre lo hecho hasta ahora, antes de seguir con más combinaciones:

1. **Aplicada (2026-10-17).** El dataset sintético no se estaba usando "infinito": en
   `Resultados/sweep.py::prepare_data`, con `data_source == "synthetic"` se llamaba a
   `generar_lote` **una sola vez** por corrida (`synthetic_train_samples`, default 384.000) y ese
   lote fijo se reusaba en todas las épocas, como si fuera un dataset normal. Corregido agregando
   un hook opcional `train_loader_fn` a `train_model` (`Modelos iniciales/common/train.py`): si se
   pasa, se llama una vez por época y su loader reemplaza a `train_loader` para esa época (sin
   `train_loader_fn` el loop es idéntico al de antes). `Resultados/sweep.py::run_one` arma ese
   hook para `data_source == "synthetic"` sin curriculum: genera un lote nuevo de
   `synthetic_train_samples` en cada época en vez de uno fijo para toda la corrida. El val
   sintético sigue generándose una sola vez por corrida (tiene que quedar fijo para que early
   stopping/scheduler comparen entre épocas). **Con curriculum + sintético sigue generándose un
   único lote fijo de train** (sin cambios): `SNRCurriculum` arma sus loaders por etapa a partir
   de ese lote y no tiene todavía una versión que regenere datos por etapa; eso es parte de la
   corrección 2, que sigue pendiente. **Las corridas de `etapa_3_sintetico_curriculum/` usan el
   comportamiento viejo (lote fijo) y quedan obsoletas** — para comparar con la nueva generación
   por época hay que rehacerlas con `--data-source synthetic` sin `--curriculum`.

2. **Aplicada (2026-10-18).** El criterio de corte del curriculum original estaba mal planteado.
   Dentro de cada etapa del curriculum (que solo entrena con un subconjunto de SNR alto), la val
   accuracy se medía sobre **todos los SNR** (`evaluate_by_snr`/`_validate` con el `val_loader`
   completo) para decidir si "ya no hay progreso" — el tutor señaló que esa métrica no es
   informativa en las primeras etapas, porque el modelo todavía no vio los SNR bajos. Se agregó
   `--curriculum-mode` (`acc` | `stage_val` | `fixed`) a `Resultados/sweep.py`, con las dos
   alternativas que proponía el tutor, **las dos probadas**:
   - **`fixed`** — épocas fijas por grupo de SNR (`--curriculum-epochs-per-stage`), sin mirar
     accuracy para decidir nada. Ver `Resultados/etapa_4_curriculum_fixed/README.md`: resuelve el
     colapso del TCN (etapa 2: 0.645 → 0.213; etapa 4: 0.632) y de las demás arquitecturas (todas
     quedan dentro de ±0.03 de la etapa 1, sin curriculum) — confirma que el problema era el
     criterio de corte, no el curriculum en sí. No se ve una mejora clara y consistente sobre no
     usar curriculum, pero es el modo que mejor se comporta de los tres.
   - **`stage_val`** — valida cada etapa solo con las muestras de val en el rango de SNR ya
     visto, y usa el val set completo solo para elegir el checkpoint final. Ver
     `Resultados/etapa_5_curriculum_stage_val/README.md`: también evita el colapso del TCN
     (0.605), pero sigue cortando cada etapa demasiado temprano (30-51 épocas vs las 100 fijas de
     `fixed`) — corregir la métrica de validación no alcanza, porque `_train_with_curriculum`
     sigue usando `patience`/`min_gain` (igual que el modo `acc`) para decidir cuándo cortar, solo
     que ahora con una señal más sana. Queda peor que `fixed` en 4 de las 5 arquitecturas.
   - `acc` (el original, con el bug) se mantiene como opción para no perder comparabilidad con
     la etapa 2, pero no se recomienda usarlo en corridas nuevas.
   Ver `SNRCurriculum`/`_train_with_curriculum` en `Modelos iniciales/common/train.py` y la
   sección "Parámetros" al principio de `Resultados/sweep.py` para el detalle de cada modo.
   **Conclusión: `fixed` es la mejor de las tres variantes probadas, pero ninguna justifica usar
   curriculum en la comparación principal del informe** (ver punto 2 de "Estado y pendientes").

Las dos correcciones del tutor (1: datos sintéticos por época; 2: criterio de corte del
curriculum) ya están aplicadas. Las etapas 2 y 3 (`Resultados/etapa_2_curriculum/`,
`Resultados/etapa_3_sintetico_curriculum/`) quedan como registro histórico del problema (criterio
`acc` / datos sintéticos de lote fijo), no como referencia para comparar — la referencia para
curriculum con RadioML real son las etapas 4 y 5.

## Al agregar una arquitectura o representación

La clase va en `common/models.py`; una transformación de datos nueva en `common/data.py`; el notebook sigue la estructura común y define su `MODEL_KWARGS`; se agrega a `MODEL_SPECS` de `comparacion/comparacion.ipynb`, a `SPECS` de `Resultados/snapshot.py` y a `ARCHS` de `Resultados/sweep.py`. Nunca copiar clases, transformaciones ni el loop de entrenamiento dentro de un notebook.

## Referencias

- `Lite_CNN.pdf`, `sensors-24-07908.pdf` — papers de referencia.
- Dataset: https://www.deepsig.ai/datasets (RadioML 2016.10a, CC BY-NC-SA 4.0).
