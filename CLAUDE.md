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
- **No commitear ni pushear** salvo pedido explícito. Hoy casi todo está sin commitear (`Modelos_Bajo_SNR/`, `Resultados/`, `CLAUDE.md` sin trackear; `*.pt` está en `.gitignore`).
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
Resultados/          etapa_0_baseline/, tcn_curriculum_fallido/ (corrida descartada), snapshot.py, sweep.py,
                     celda_guardar_corrida.py, sweep/ (corridas del barrido y de notebooks)
Datos_Sinteticos/    generador.py (señales I/Q sintéticas, numpy/scipy), validar.py, README.md — NO conectado a notebooks
```

### `Modelos iniciales/common/` (única fuente de verdad; ningún notebook redefine modelos, datos ni loop)

- `models.py` — `CNNBaseline`, `LSTMClassifier`, `GRUClassifier`, `TCN` (+`ResidualBlock`), `ULCNN` (+`ComplexConv1d`, `IQCF`, `ChannelShuffle`, `ChannelAttention`, `FMDR`). Configurables por kwargs (`channels`, `hidden_size`, `num_layers`, `dilations`, `n_cv`, `n_fmdr`...); los defaults reproducen las arquitecturas con pesos ya entrenados. `LSTMClassifier`/`GRUClassifier` tienen `input_norm` (default `False`, `BatchNorm1d` sobre I/Q antes de la recurrencia): **sin normalizar, LSTM/GRU casi no aprenden** (señal ≈ 0.005) — CNN/TCN se salvan porque tienen BatchNorm tras la primera conv. `ULCNN` tiene ~9k parámetros (vs 40k–1M del resto).
- `data.py` — `load_radioml_digital`, `split_dataset` (60/20/20 estratificado, seed 42, mezcla previa porque el dataset viene ordenado por clase), `make_loader`, `to_amplitude_phase`, `augment_rotate_iq` (rotación de constelación π/2, π, 3π/2 → train 4x; solo en train, y **antes** de pasar a A/φ).
- `train.py` — `train_model` (Adam + `EarlyStopping` sobre val loss; `use_lr_scheduler=True` agrega `ReduceLROnPlateau`; `curriculum=` activa el entrenamiento por grupos de SNR), `SNRCurriculum`, `evaluate_by_snr`, `overall_accuracy`, `compute_cm`. Sin curriculum, el camino es idéntico al original (verificado época a época).

### Curriculum por SNR (`SNRCurriculum`, `USE_CURRICULUM = True` por defecto en los 6 notebooks de `Modelos iniciales/`)

Idea del usuario/tutor: entrenar de SNR alto a bajo y cortar cuando ya no hay progreso (las muestras de SNR muy bajo, ~30% del dataset, dan gradientes poco informativos). Reglas: los 20 niveles de SNR se agrupan de a `group_size=4` (5 etapas); cada etapa **suma** el grupo siguiente (más bajo) y entrena hasta `patience=3` épocas sin mejorar la **val accuracy sobre todos los SNR** en más de `min_gain=0.005` (o `max_epochs_per_stage=15`); si una etapa no mejora la mejor accuracy, se corta y se devuelven los mejores pesos. Se usa accuracy y no loss porque al sumar SNR muy bajos la loss puede mejorar solo por calibración. Guarda de arranque: mientras la mejor accuracy no supere el azar + 0.05 no se cuenta "sin progreso" (tope `max_warmup_epochs=30`, con aviso) — sin esto el curriculum cortó LSTM/GRU a las 7 épocas con un modelo que no había aprendido nada. En este modo `PATIENCE` no se usa y `EPOCHS=80` es solo tope. `history.pkl` guarda `curriculum`, `curriculum_stage` y `curriculum_min_snr` por época. El checkpoint se elige por val accuracy (el baseline `etapa_0` lo eligió por val loss). `cnn_preproc` NO está conectado al curriculum (tiene su propia receta: scheduler, 200 épocas, batch 128; en `sweep.py` se reproduce con `--scheduler 1 --batch-sizes 128 --epochs 200`).

### Notebooks de `Modelos iniciales/` (cada uno entrena UN modelo, importa todo de `common/`)

Estructura común: imports/config (`MODEL_KWARGS`, `USE_CURRICULUM`...) → datos (con `augment_rotate_iq`) → entrenamiento → evaluación → gráficos (curva de pérdida, accuracy vs SNR, matriz de confusión global y por SNR <0 / >0 dB). Guardan `<nombre>.pt` + `history.pkl` (con `model_kwargs`) en su carpeta. `cnn_ap`/`tcn_ap` = mismas `CNNBaseline`/`TCN` sobre A/φ. `cnn_preproc` = `ULCNN` con su receta original. Config actual notable: `gru` `hidden_size=64, fc_hidden=32, dropout=0.4, batch 64, lr 1e-3, input_norm=True`; `lstm` `hidden_size=128, input_norm=True, lr 3e-4` (con 1e-3 la LSTM es errática; con 1e-4 no despega); `tcn` `channels=64, dilations=(1,2,4), dropout=0.4` (**repuesto** a la config del baseline `etapa_0`, 75.592 params; el default de la clase es `(1,2,4,8)`). `comparacion/comparacion.ipynb` no entrena: reconstruye cada modelo desde su `history.pkl` (`model_kwargs`) y lo evalúa con su representación; `MODEL_SPECS` lista los 7 modelos. `Modelos iniciales/` tiene `.pt` e `history.pkl` para los 7; `cnn` y `cnn_ap` tienen el `.pt` del 2026-09-07 (anterior a augmentation/curriculum) y **hay que reentrenarlos**.

### `Modelos_Bajo_SNR/` (estrategias para SNR ≤ -10 dB; mismo patrón de notebooks)

`common_snr/` se llama así (no `common`) para no chocar con `Modelos iniciales/common` cuando ambos están en `sys.path`; los notebooks importan `common.data`/`common.train` de `../../Modelos iniciales` y solo agregan lo específico: `features.py` (`compute_hos_features` — cumulantes C40/C42/C63, validados: BPSK da C40=-2 —, `compute_stft`, `compute_cwt` con PyWavelets, `pack_fusion_features`) y `models.py` (`HOSClassifier`, `STFTCNN` [sirve para STFT y wavelet], `CNNLSTMHybrid`, `SpectrogramULCNN` [ULCNN 2D], `FusionClassifier` [ramas I/Q+HOS+STFT; recibe un único tensor plano y lo desempaqueta, para poder reusar `train_model` sin cambios]). Notebooks: `hos/`, `stft/`, `wavelet/`, `lstm_snr/`, `stft_ulcnn/`, `fusion/`, `comparacion/`. Pendientes de la lista original (README de la carpeta): ciclostacionariedad, transformers, autoencoders, aprendizaje contrastivo. **Desactualizado**: `Modelos_Bajo_SNR/README.md` y `comparacion/` no incluyen `stft_ulcnn` ni `fusion` (README tampoco `wavelet`).

### `Resultados/` — guardar etapas y barridos sin pisar nada

- `etapa_0_baseline/` — snapshot de los 13 modelos (7 de `Modelos iniciales` + 6 de `Modelos_Bajo_SNR`) **antes** de augmentation/curriculum: `.pt`, `history.pkl`, `.png`, notebook (los de `Modelos iniciales` tomados de git HEAD `b9f488c`) y `metricas.json`.
- `snapshot.py <etapa> [--grupos "Modelos iniciales"] [--sobrescribir]` — copia + recalcula métricas en el test set. Aborta si la etapa ya existe. **No descubre modelos nuevos solos**: cada uno se registra en `SPECS` (clase, `.pt`, representación); avisa con `SIN SPEC` si encuentra carpetas no registradas. Próximo paso previsto: tras reentrenar, `python Resultados/snapshot.py etapa_1_augmentation_curriculum --grupos "Modelos iniciales"`.
- `sweep.py` — barrido independiente de los notebooks; cada corrida en `Resultados/sweep/<id>/` (`config.json` completo con `data_source`, `lr_scheduler`, etc.; `metrics.json`, `history.json`, `model.pt`), nunca pisa (salta lo ya hecho), `--dry-run` muestra el plan, `--params 10000 50000 base` compara arquitecturas a **igual cantidad de parámetros** (busca el ancho que se acerque; CNN escala también su capa densa), `--scheduler 0 1` activa `ReduceLROnPlateau` (0.8, patience 10), `--resumen` arma `resumen.csv` + gráfico accuracy vs parámetros. Cubre `cnn`, `lstm`, `gru`, `tcn` y `ulcnn` (`cnn_ap`/`tcn_ap` = `--repr ap`); los 6 de `Modelos_Bajo_SNR` no están. Los `base` de `ARCHS` son constantes escritas a mano: mantenerlos alineados con los notebooks. Hay 11 corridas guardadas (incluidas algunas fallidas de `tcn` con curriculum y una con nombre raro `tcn_iq_ap_iq_p79616`, sin revisar); `data_source` es siempre `radioml` (no hay opción sintética todavía).
- `celda_guardar_corrida.py` — celda genérica para pegar al final de cada notebook: guarda esa corrida en `Resultados/sweep/` con el mismo formato (nombre con fecha, no pisa). Hay que editar `LR`, `AUGMENT`, `SCHEDULER` (`True` en `cnn_preproc`) y `DATA_CONFIG` (`{"data_source": "radioml"}` por defecto; si se entrenó con datos sintéticos, `generador.config(...)`) a mano; el resto sale de las variables del notebook. `sweep.py --resumen` junta corridas de notebooks y del barrido.

## Estado y pendientes (feedback del tutor, 5 puntos)

1. **Augmentation pareja** (comparar CNN vs `cnn_preproc` sin confundir preprocesamiento con arquitectura): aplicada a los 6 notebooks de `Modelos iniciales/`; **hay que reentrenarlos** (el usuario los está corriendo).
2. **Curriculum por SNR**: implementado y activo (ver arriba); pendiente de resultados reales.
3. **Datos sintéticos on-the-fly** (generar señales con numpy/scipy + AWGN a distintos SNR, dataset "infinito", RadioML solo para test; GNU Radio se descartó como dependencia): **generador listo** en `Datos_Sinteticos/generador.py` (`generar_lote`, `generador_infinito`, `config()` con `VERSION`), validado (SNR medido ±0.02 dB). **Sin conectar** a notebooks ni al barrido; no tiene canal con fading ni offset de reloj como RadioML.
4. **Sweep de hiperparámetros/capacidad** (lr, batch, tamaño; sobre todo mismo #parámetros entre arquitecturas, p. ej. ULCNN con la capacidad de TCN): herramienta lista (`sweep.py`), con algunas corridas hechas (ver arriba), sin barrido completo ni análisis. `history.pkl` histórico no guarda `lr`/`batch_size` (los del baseline se reconstruyen de los notebooks).
5. **I/Q + A/φ juntos** (dos ramas de features + cabeza común): **no empezado**; reusar el patrón de `FusionClassifier`.

Resultados de `etapa_0` (accuracy global / a SNR ≤ -10 dB, test): TCN A/φ 0.653/0.168 · ULCNN 0.616/0.161 (9.364 params) · CNN+LSTM (lstm_snr) 0.595/0.164 · TCN 0.576/0.144 · CNN A/φ 0.521/0.141 · CNN 0.513/0.144 · HOS 0.442/0.128 · GRU 0.411/0.141 · STFT 0.231 · fusion 0.196 · LSTM 0.179 · wavelet 0.177 · STFT+ULCNN2D 0.123. **A SNR ≤ -10 dB todos están en el azar (≈0.125)** y varios modelos de `Modelos_Bajo_SNR` están cerca del azar también en accuracy global (probablemente no convergieron; revisar, y ver si les afecta la escala de la señal como a LSTM/GRU). Los nuevos GRU/LSTM (`input_norm`, otro lr) no son comparables con `etapa_0`, cuya versión estaba rota: contarlos como su propia etapa.

## Al agregar una arquitectura o representación

La clase va en `common/models.py` (o `common_snr/models.py` si es de bajo SNR); una transformación de datos nueva en `common/data.py` (o `common_snr/features.py`); el notebook sigue la estructura común y define su `MODEL_KWARGS`; se agrega a `MODEL_SPECS` de la `comparacion` correspondiente, a `SPECS` de `Resultados/snapshot.py` y a `ARCHS` de `Resultados/sweep.py`. Nunca copiar clases, transformaciones ni el loop de entrenamiento dentro de un notebook.

## Referencias

- `Lite_CNN.pdf`, `sensors-24-07908.pdf` — papers de referencia.
- Dataset: https://www.deepsig.ai/datasets (RadioML 2016.10a, CC BY-NC-SA 4.0).
