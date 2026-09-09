# AMC_CEIA — Clasificación Automática de Modulaciones (AMC)

Proyecto final de la Carrera de Especialización en Inteligencia Artificial (CEIA - FIUBA), enfocado en la clasificación automática de modulaciones de señales de radio utilizando el dataset **RadioML**.

Se entrenaron y compararon distintas arquitecturas de deep learning (CNN, LSTM, GRU, TCN y ULCNN) para identificar el tipo de modulación de una señal de radio a partir de sus componentes I/Q (y, en algunas variantes, de su representación Amplitud/Fase), evaluando el desempeño en distintas condiciones de SNR (relación señal-ruido). El objetivo no es solo maximizar accuracy, sino encontrar el modelo **más liviano** que cumpla el objetivo, pensando en un futuro despliegue embebido/tiempo real.

## Estructura del repositorio

```
├── Modelos iniciales/
│   ├── rfml/              # Modelo CNN base (librería rfml) + análisis exploratorio (EDA) + datos
│   ├── common/             # Arquitecturas, carga de datos y loop de entrenamiento compartidos
│   ├── cnn/                # Entrenamiento del modelo CNN (I/Q)
│   ├── lstm/                # Entrenamiento del modelo LSTM (I/Q)
│   ├── gru/                 # Entrenamiento del modelo GRU (I/Q)
│   ├── tcn/                  # Entrenamiento del modelo TCN (I/Q)
│   ├── cnn_ap/                # CNN, pero con representación Amplitud/Fase en vez de I/Q
│   ├── tcn_ap/                 # TCN, pero con representación Amplitud/Fase en vez de I/Q
│   ├── cnn_preproc/             # ULCNN — arquitectura ultra-liviana (~9k parámetros)
│   └── comparacion/               # Compara todos los modelos ya entrenados (no reentrena)
├── requirements.txt       # Dependencias del proyecto
├── Lite_CNN.pdf            # Paper de referencia
├── sensors-24-07908.pdf    # Paper de referencia
└── README.md
```

Cada carpeta de arquitectura contiene un notebook que entrena **un solo modelo** (importando arquitectura, datos y loop de entrenamiento desde `common/`), sus pesos entrenados (`.pt`) y su historial de entrenamiento (`history.pkl`). `comparacion/` solo carga esos pesos y genera los gráficos comparativos (accuracy vs SNR, curvas de pérdida, matrices de confusión — globales y por rango de SNR —, accuracy vs parámetros).

Ver [`Modelos iniciales/README.md`](Modelos%20iniciales/README.md) para el detalle de cada carpeta, cómo funciona `common/`, y cómo agregar una arquitectura nueva.

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/manuel-dieguezw/AMC_CEIA.git
cd AMC_CEIA
```

2. Crear y activar un entorno virtual:
```bash
python -m venv amc_env
# Windows
.\amc_env\Scripts\activate
# Linux/Mac
source amc_env/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Datos

Los notebooks requieren dos archivos de datos que **no están versionados en el repositorio** por su tamaño (más de 100 MB):

- `radioml_X.npy`
- `radioml_X_digital.npy`

Se pueden descargar desde el siguiente Drive:

📁 **[Descargar datos (Google Drive)](https://drive.google.com/drive/folders/1Ron4S_tnuyWSMxGtOiSAmvr7B-ChwYw_?usp=sharing)**

Una vez descargados, colocarlos en:
```
Modelos iniciales/rfml/radioml_X.npy
Modelos iniciales/rfml/radioml_X_digital.npy
```

Los archivos de metadata (`radioml_metadata.parquet`, `radioml_metadata_digital.parquet`) sí están incluidos en el repositorio.

### Dataset original

Los `.npy` de este repositorio fueron generados a partir del dataset público **RadioML 2016.10a**, publicado por DeepSig. Es un dataset sintético generado con GNU Radio, con 11 tipos de modulación (8 digitales, 3 analógicas) en un rango de SNR de -20dB a 18dB.

- Fuente oficial: [https://www.deepsig.ai/datasets](https://www.deepsig.ai/datasets)
- Descarga directa: `http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2`
- Licencia: Creative Commons Attribution - NonCommercial - ShareAlike 4.0 (CC BY-NC-SA 4.0)

El archivo original está en formato pickle (`RML2016.10a_dict.pkl`); el preprocesamiento a `.npy` y `.parquet` se realiza en `EDA_inicial.ipynb`.

## Uso

Abrir los notebooks en el siguiente orden sugerido:

1. `Modelos iniciales/rfml/EDA_inicial.ipynb` — análisis exploratorio del dataset
2. `Modelos iniciales/rfml/rfml.ipynb` — modelo CNN base (librería `rfml`)
3. `Modelos iniciales/cnn/cnn.ipynb`, `lstm/lstm.ipynb`, `gru/gru.ipynb`, `tcn/tcn.ipynb`, `cnn_ap/cnn_ap.ipynb`, `tcn_ap/tcn_ap.ipynb`, `cnn_preproc/cnn_preproc.ipynb` — entrenamiento de cada arquitectura (en cualquier orden)
4. `Modelos iniciales/comparacion/comparacion.ipynb` — comparación entre las arquitecturas entrenadas

## Referencias

- Lite_CNN.pdf
- sensors-24-07908.pdf
