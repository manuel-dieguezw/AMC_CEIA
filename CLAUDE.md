# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AMC_CEIA — proyecto final de la Carrera de Especialización en Inteligencia Artificial (CEIA - FIUBA). Clasificación automática de modulaciones (AMC) a partir de las componentes I/Q de señales de radio, con el dataset **RadioML 2016.10a** (subset de 8 modulaciones digitales, SNR de -20 a 18 dB). Se comparan arquitecturas de deep learning (CNN, LSTM, GRU, TCN, ULCNN) y estrategias para bajo SNR, evaluando accuracy en función del SNR.

**Objetivo real**: no es solo maximizar accuracy — es encontrar el modelo **más liviano** que cumpla el objetivo, pensando en despliegue en tiempo real / embebidos. Priorizar también parámetros, FLOPs, latencia y tamaño, no solo accuracy vs SNR.

El trabajo vive en notebooks Jupyter; no hay app ni paquete instalable (aparte del venv `amc_env/` y la librería externa `rfml`). El usuario escribe en español rioplatense; responder en español.

## Reglas de trabajo con este usuario (leer primero)

- **Tocar solo lo que el usuario nombra.** No regenerar notebooks enteros ni hacer pasadas amplias "por consistencia": una regeneración temprana borró los outputs guardados de `gru.ipynb` (irrecuperable, nunca estuvo en git). Editar notebooks solo con `NotebookEdit` (celda puntual, preserva los outputs del resto).
- **Si el usuario está corriendo notebooks, no editarlos**: darle la celda para que la pegue él (así se hizo con `Resultados/celda_guardar_corrida.py`). Antes de editar un notebook hay que leerlo con `Read` (la herramienta lo exige y los ids de celda cambian: Jupyter los regenera al guardar, p. ej. `gru.ipynb` tiene ids tipo `e8424eb7`, `lstm.ipynb` usa `cell-N`).
- **Jupyter cachea los módulos**: si se cambia algo en `Modelos iniciales/common/` mientras hay un kernel abierto, el notebook sigue usando la versión vieja (error típico: `unexpected keyword argument`). Hay que reiniciar el kernel.
- **No lanzar entrenamientos largos** (CPU, dataset aumentado 4x: cada época tarda minutos). Validar con subconjuntos chicos, pocas épocas y una carpeta temporal `Modelos iniciales/_smoke/` (hermana de `rfml/` para que `../rfml` resuelva) que se borra al terminar. Cuando el código de una celda se prueba, ejecutar el código real del notebook (leer el `.ipynb` con `json`, `exec` de las celdas), no una copia.
- **No commitear ni pushear** salvo pedido explícito. `Modelos_Bajo_SNR/`, `amc_env/` y `*.pt` están en `.gitignore` y nunca quedaron trackeados, así que no aparecen en `git status`. `CLAUDE.md` sí está en `.gitignore` pero quedó trackeado de antes de agregarlo (sigue apareciendo como modificado); no destrackearlo sin pedido explícito.
- Los `.png`/`history.pkl`/`.pt` de cada notebook se **sobrescriben** al reentrenar. Antes de reentrenar una etapa, hacer snapshot (ver `Resultados/`).
- Shell en Windows/bash: heredocs con comillas complicadas fallan; si un comando es largo, escribir el script a un archivo y ejecutarlo. Los scripts que cargan datos deben correr con cwd = carpeta hermana de `rfml/` o pasar rutas absolutas (`load_radioml_digital(dataset_folder=...)`).

## Setup y comandos

```bash
.\amc_env\Scripts\activate        # Windows (venv ya existe en amc_env/)
pip install -r requirements.txt   # incluye PyWavelets (requirements.txt regenerado en UTF-8 con pip freeze)
```

No hay build, lint ni tests: "run" = ejecutar celdas de notebook o los scripts de `Resultados/`. Python del venv: `D:/CEIA/Proyecto_AMC/amc_env/Scripts/python.exe`.

### Datos

`radioml_X.npy` / `radioml_X_digital.npy` no están versionados (>100 MB; ver README para descargarlos) y viven en `Modelos iniciales/rfml/`, junto con los `.parquet` de metadata. Los notebooks los leen por ruta relativa `../rfml/` (desde `Modelos iniciales/<x>/`) o `../../Modelos iniciales/rfml` (desde `Modelos_Bajo_SNR/<x>/`). **La señal tiene amplitud minúscula (std ≈ 0.006)**; el modelo nunca recibe el SNR como entrada (solo I/Q o A/φ) — el SNR se usa solo para evaluar y, en el curriculum, para elegir qué muestras de train entran.

## Estructura del repo

```
Modelos iniciales/   rfml/ (baseline con lib externa + datos), common/ (código compartido),
                     cnn/ lstm/ gru/ tcn/ cnn_ap/ tcn_ap/ cnn_preproc/ (un notebook por modelo), comparacion/
Modelos_Bajo_SNR/    common_snr/ + hos/ stft/ wavelet/ lstm_snr/ stft_ulcnn/ fusion/ comparacion/
Resultados/          etapa_0_baseline/ .. etapa_3_sintetico_curriculum/ (basadas en sweep.py, con README propio),
                     etapa_0_original/ y etapa_1_augmentation_curriculum/ (legacy, basadas en snapshot.py),
                     tcn_curriculum_fallido/ (corrida descartada), snapshot.py, sweep.py, celda_guardar_corrida.py,
                     sweep/ (corridas del barrido y de notebooks), comparar_barrido.ipynb, comparar_etapas.ipynb
Datos_Sinteticos/    generador.py (señales I/Q sintéticas, numpy/scipy), validar.py, README.md — conectado a
                     Resultados/sweep.py (--data-source synthetic); NO conectado a los notebooks
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

### `Modelos_Bajo_SNR/` (estrategias para SNR ≤ -10 dB; mismo patrón de notebooks)

`common_snr/` se llama así (no `common`) para no chocar con `Modelos iniciales/common` cuando ambos están en `sys.path`; los notebooks importan `common.data`/`common.train` de `../../Modelos iniciales` y solo agregan lo específico: `features.py` (`compute_hos_features` — cumulantes C40/C42/C63, validados: BPSK da C40=-2 —, `compute_stft`, `compute_cwt` con PyWavelets, `pack_fusion_features`) y `models.py` (`HOSClassifier`, `STFTCNN` [sirve para STFT y wavelet], `CNNLSTMHybrid`, `SpectrogramULCNN` [ULCNN 2D], `FusionClassifier` [ramas I/Q+HOS+STFT; recibe un único tensor plano y lo desempaqueta, para poder reusar `train_model` sin cambios]). Notebooks: `hos/`, `stft/`, `wavelet/`, `lstm_snr/`, `stft_ulcnn/`, `fusion/`, `comparacion/`. Pendientes de la lista original (README de la carpeta): ciclostacionariedad, transformers, autoencoders, aprendizaje contrastivo. **Desactualizado**: `Modelos_Bajo_SNR/README.md` y `comparacion/` no incluyen `stft_ulcnn` ni `fusion` (README tampoco `wavelet`).

### `Resultados/` — guardar etapas y barridos sin pisar nada

Ver `Resultados/README.md` para el detalle completo (cómo leer el nombre de una corrida, qué
archivos tiene cada una, etc.). Resumen:

- **Etapas basadas en `sweep.py` (la referencia actual)**: `etapa_0_baseline/` (sin augmentation/scheduler/curriculum) → `etapa_1_augmentation_scheduler/` (+ ambos) → `etapa_2_curriculum/` (+ curriculum, RadioML real) → `etapa_3_sintetico_curriculum/` (datos sintéticos + curriculum, test RadioML real). Cada una tiene `metricas.json` (armado a mano desde las corridas de `sweep/` citadas en `fuente`) y su propio `README.md` con el comando, la tabla y el análisis. `Resultados/comparar_etapas.ipynb` las compara entre sí.
- **Etapas basadas en `snapshot.py` (legacy)**: `etapa_0_original/` (13 modelos, 7 de `Modelos iniciales` + 6 de `Modelos_Bajo_SNR`, antes de augmentation/curriculum) y `etapa_1_augmentation_curriculum/`. Quedaron de respaldo; no son la referencia para comparar ahora, porque otro camino (los notebooks sin reentrenar con la receta final) las dejó desactualizadas.
- `snapshot.py <etapa> [--grupos "Modelos iniciales"] [--sobrescribir]` — copia `.pt`/`history.pkl`/notebook de las carpetas de los modelos + recalcula métricas en el test set. Aborta si la etapa ya existe. **No descubre modelos nuevos solos**: cada uno se registra en `SPECS`; avisa con `SIN SPEC` si encuentra carpetas no registradas.
- `sweep.py` — barrido independiente de los notebooks; cada corrida en `Resultados/sweep/<id>/` (`config.json` completo con `data_source`, `lr_scheduler`, etc.; `metrics.json`, `history.json`, `model.pt`), nunca pisa (salta lo ya hecho), `--dry-run` muestra el plan, `--params 10000 50000 base` compara arquitecturas a **igual cantidad de parámetros**, `--scheduler 0 1` activa `ReduceLROnPlateau` (0.8, patience 10), `--curriculum 0 1` activa `SNRCurriculum` (se puede combinar con `--scheduler 1`), `--data-source radioml synthetic` (sintético: train/val on-the-fly con `Datos_Sinteticos/generador.py`, test siempre RadioML real; `--synthetic-train-samples`/`--synthetic-val-samples`/`--synthetic-escala`; `augment` se ignora con datos sintéticos), `--resumen` arma `resumen.csv` + gráfico accuracy vs parámetros. Cubre `cnn`, `lstm`, `gru`, `tcn` y `ulcnn` (`cnn_ap`/`tcn_ap` = `--repr ap`); los 6 de `Modelos_Bajo_SNR` no están. Los `base` de `ARCHS` son constantes escritas a mano: mantenerlos alineados con los notebooks. Hay más de 40 corridas guardadas, de barridos distintos (lr, capacidad, curriculum, sintético, algunas fallidas/duplicadas de pruebas); usar `--resumen` o `comparar_barrido.ipynb` para filtrar, no asumir que todas son comparables entre sí.
- `celda_guardar_corrida.py` — plantilla genérica para notebooks nuevos (guarda la corrida en `Resultados/sweep/`, mismo formato, no pisa). En los 7 notebooks de `Modelos iniciales/` ya está adaptada e integrada: lee `LR`/`AUGMENT`/`SCHEDULER`/`USE_CURRICULUM` directamente de las celdas de arriba, no hay que editarla a mano.

## Estado y pendientes (feedback del tutor, 5 puntos)

1. **Augmentation pareja** (comparar CNN vs `cnn_preproc` sin confundir preprocesamiento con arquitectura): aplicada a los 7 notebooks de `Modelos iniciales/` como flag (`AUGMENT`); la comparación real a igual receta vive en las etapas de `sweep.py` (0 a 3, ver arriba), que ya la tienen pareja. **Los notebooks siguen sin reentrenarse con la receta final** (lr unificado, `tcn`/`tcn_ap` con la misma arquitectura).
2. **Curriculum por SNR**: implementado, combinable con el scheduler, y con resultados reales (ver arriba): **empeora las 5 arquitecturas y colapsa al TCN con RadioML real**; con datos sintéticos el TCN no colapsa. Conclusión para el informe: no usarlo en la comparación principal.
3. **Datos sintéticos on-the-fly**: generador listo y **conectado a `Resultados/sweep.py`** (`--data-source synthetic`), con resultados reales en `etapa_3_sintetico_curriculum/` (y la variante sin curriculum, si se corrió — ver `Resultados/sweep/`). **Sin conectar a los notebooks**. No tiene canal con fading ni offset de reloj como RadioML; eso probablemente explica por qué rinde algo peor que RadioML real salvo en el TCN.
4. **Sweep de hiperparámetros/capacidad**: herramienta lista y usada para armar las etapas 0-3 y el barrido de lr (`Resultados/sweep/README.md`). **Falta el barrido a igual número de parámetros** (`--params 10000 50000`, comparar ULCNN con la capacidad de TCN, etc.) — todavía no se corrió.
5. **I/Q + A/φ juntos** (dos ramas de features + cabeza común): **no empezado**; reusar el patrón de `FusionClassifier`.

Resultados de `etapa_0_original` (legacy, 13 modelos, accuracy global / a SNR ≤ -10 dB, test): TCN A/φ 0.653/0.168 · ULCNN 0.616/0.161 (9.364 params) · CNN+LSTM (lstm_snr) 0.595/0.164 · TCN 0.576/0.144 · CNN A/φ 0.521/0.141 · CNN 0.513/0.144 · HOS 0.442/0.128 · GRU 0.411/0.141 · STFT 0.231 · fusion 0.196 · LSTM 0.179 · wavelet 0.177 · STFT+ULCNN2D 0.123. **A SNR ≤ -10 dB todos están en el azar (≈0.125)**. Estos números son de la receta original (sin augmentation/scheduler/curriculum) y los de `Modelos_Bajo_SNR`; para los 5 principales (`cnn`, `lstm`, `gru`, `tcn`, `ulcnn`) la referencia actual son las etapas 0-3 basadas en `sweep.py` (`Resultados/etapa_*/README.md`), no estos valores.

## Al agregar una arquitectura o representación

La clase va en `common/models.py` (o `common_snr/models.py` si es de bajo SNR); una transformación de datos nueva en `common/data.py` (o `common_snr/features.py`); el notebook sigue la estructura común y define su `MODEL_KWARGS`; se agrega a `MODEL_SPECS` de la `comparacion` correspondiente, a `SPECS` de `Resultados/snapshot.py` y a `ARCHS` de `Resultados/sweep.py`. Nunca copiar clases, transformaciones ni el loop de entrenamiento dentro de un notebook.

## Referencias

- `Lite_CNN.pdf`, `sensors-24-07908.pdf` — papers de referencia.
- Dataset: https://www.deepsig.ai/datasets (RadioML 2016.10a, CC BY-NC-SA 4.0).
