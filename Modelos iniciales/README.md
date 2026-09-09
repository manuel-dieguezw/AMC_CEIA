# Modelos iniciales

Notebooks de exploración y entrenamiento para la clasificación automática de modulaciones (AMC)
sobre RadioML 2016.10a. Cada arquitectura vive en su propia carpeta, entrena un solo modelo, y
`comparacion/` es el único notebook que junta todo para comparar.

## Estructura

```
Modelos iniciales/
├── rfml/            # Baseline con la librería externa rfml + EDA + datos (.npy, .parquet)
├── common/          # Código compartido: arquitecturas, carga de datos, loop de entrenamiento
├── cnn/             # Entrena CNNBaseline (I/Q)
├── lstm/            # Entrena LSTMClassifier (I/Q)
├── gru/             # Entrena GRUClassifier (I/Q)
├── tcn/             # Entrena TCN (I/Q)
├── cnn_ap/          # CNNBaseline, pero con representación Amplitud/Fase en vez de I/Q
├── tcn_ap/          # TCN, pero con representación Amplitud/Fase en vez de I/Q
├── cnn_preproc/     # ULCNN — arquitectura ultra-liviana (conv. compleja + depthwise/attention)
└── comparacion/     # No entrena nada: carga todos los .pt y compara
```

## Orden de ejecución

1. **`rfml/EDA_inicial.ipynb`** — convierte el pickle original de RadioML (`RML2016.10a_dict.pkl`)
   a `radioml_X.npy` / `radioml_metadata.parquet` (y las variantes `_digital`, filtradas a las 8
   modulaciones digitales que usa el resto de los notebooks). Los `.npy` no están versionados por
   tamaño — ver el README raíz del repo para descargarlos.
2. **`rfml/rfml.ipynb`** — CNN baseline con la librería `rfml`. Es un caso aparte, no comparte
   código con el resto.
3. **Cualquier orden**: `cnn/cnn.ipynb`, `lstm/lstm.ipynb`, `gru/gru.ipynb`, `tcn/tcn.ipynb`,
   `cnn_ap/cnn_ap.ipynb`, `tcn_ap/tcn_ap.ipynb`, `cnn_preproc/cnn_preproc.ipynb` — cada uno
   entrena su modelo y guarda `<nombre>.pt` + `history.pkl` en su propia carpeta.
4. **`comparacion/comparacion.ipynb`** — al final, una vez entrenados los modelos que quieras
   comparar. Carga los `.pt`/`history.pkl` correspondientes y genera accuracy vs SNR, curvas de
   pérdida, matrices de confusión (globales y por rango de SNR) y accuracy vs cantidad de
   parámetros.

## `common/` — código compartido

Para no duplicar arquitecturas ni loops de entrenamiento entre notebooks:

- **`models.py`** — `CNNBaseline`, `LSTMClassifier`, `GRUClassifier`, `TCN` y `ULCNN`. Todas
  configurables vía argumentos del constructor (profundidad/ancho) — los defaults reproducen la
  arquitectura con la que ya hay pesos entrenados.
- **`data.py`** — `load_radioml_digital` (carga + shuffle), `split_dataset` (60/20/20
  estratificado), `make_loader`, `to_amplitude_phase` (I/Q → Amplitud/Fase), `augment_rotate_iq`
  (data augmentation por rotación de constelación, usada por `cnn_preproc/`).
- **`train.py`** — `train_model` (con `EarlyStopping` y `ReduceLROnPlateau` opcional),
  `evaluate_by_snr`, `overall_accuracy`, `compute_cm`.

## Cada notebook de entrenamiento

Todos siguen la misma estructura: imports/configuración → carga de datos → entrenamiento →
evaluación → gráficos. La celda de configuración define `MODEL_KWARGS` — para probar otra
profundidad/ancho de la arquitectura, se edita ese diccionario y se vuelve a correr el notebook.
La config usada queda guardada en `history.pkl` junto con el historial de entrenamiento, así
`comparacion.ipynb` reconstruye cada modelo con los parámetros exactos antes de cargar los pesos.

## Notas sobre el objetivo del proyecto

El objetivo no es solo maximizar accuracy — es encontrar el modelo **más liviano** que cumpla el
objetivo de clasificación, pensando en un futuro despliegue embebido/tiempo real. Por eso
`comparacion.ipynb` reporta también cantidad de parámetros, y `cnn_preproc/` (ULCNN, ~9 mil
parámetros vs >100 mil del resto) es el candidato más prometedor si sostiene accuracy.

## Agregar un modelo nuevo

1. La clase de arquitectura va en `common/models.py`.
2. Si necesita una transformación de datos nueva, va en `common/data.py`.
3. El notebook de entrenamiento sigue la misma estructura que los existentes (usarlos como
   plantilla) y define su propio `MODEL_KWARGS`.
4. Se agrega su entrada en `MODEL_SPECS` dentro de `comparacion.ipynb`.

Nunca copiar una clase de modelo, una transformación de datos o el loop de entrenamiento dentro
de un notebook — siempre importar desde `common/`.
