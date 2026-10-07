# Modelos_Bajo_SNR

Estrategias específicas para mejorar la clasificación automática de modulaciones (AMC) en el
rango de bajo SNR, donde los modelos de `../Modelos iniciales/` (incluido `ULCNN`, el más
liviano) caen a precisiones cercanas al azar.

## El problema fundamental

Por debajo de -10 dB la señal tiene menos energía que el ruido, así que cualquier
característica que se intente extraer está corrompida. Existe además un límite físico de
información de Fisher que ningún modelo puede superar independientemente de su arquitectura:
por debajo de cierto SNR la señal simplemente no contiene suficiente información para
distinguir modulaciones con certeza. En la práctica ese umbral está alrededor de -10 a -14 dB
dependiendo de la modulación.

## Estrategias implementadas (primera etapa)

| Carpeta | Estrategia | Representación de entrada | Modelo |
|---|---|---|---|
| `hos/` | Momentos/cumulantes de orden superior | Vector de 16 features (`common_snr/features.py::compute_hos_features`) | `HOSClassifier` (MLP chico) |
| `stft/` | Distribución tiempo-frecuencia (STFT) | Espectrograma complejo, 2 canales (`compute_stft`) | `STFTCNN` (CNN 2D) |
| `lstm_snr/` | Memoria temporal larga (estilo MCLDNN) | I/Q crudo | `CNNLSTMHybrid` (CNN + LSTM) |
| `comparacion/` | — | — | Compara las 3 de arriba entre sí y contra baselines de `Modelos iniciales/` (CNN, TCN, ULCNN) |

**Por qué estas 3 primero**: son las más rápidas de implementar y validar con lo que ya existe
en el repo, y cubren tres ángulos distintos (feature engineering estadístico, representación
alternativa de la señal, arquitectura con memoria) sin requerir infraestructura nueva grande.

### Pendiente para una segunda etapa

- **Ciclostacionariedad (SCDF/CAF)**: espectro de densidad espectral cíclica — sólida base
  teórica y robusta a muy bajo SNR, pero costosa de calcular y requiere más trabajo de
  validación que HOS/STFT.
- **Transformers con atención global**: pueden relacionar muestras muy separadas en el tiempo,
  pero son costosos computacionalmente y necesitan más datos para entrenar bien — no calzan
  con el objetivo de liviandad sin antes probar si vale la pena el trade-off.
- **Autoencoders para representación robusta al ruido**: preentrenamiento no supervisado
  separando señal de ruido antes de clasificar. Útil si en algún momento hay señales sin
  etiquetar disponibles.
- **Aprendizaje contrastivo**: el enfoque más prometedor en la literatura reciente (invariancia
  al SNR por diseño), pero requiere un diseño cuidadoso de pares positivos/negativos — se deja
  para después de ver si las estrategias más simples ya mueven la aguja.

### Comparativa completa (incluye lo pendiente)

| Estrategia | Ventaja principal | Inconveniente principal |
|---|---|---|
| Ciclostacionariedad | Sólida base teórica, robusta a muy bajo SNR | Costosa computacionalmente, requiere conocimiento previo |
| Tiempo-frecuencia | Intuitiva, compatible con CNN 2D estándar | Resolución tiempo-frecuencia limitada por incertidumbre de Heisenberg |
| Momentos de orden superior | Muy robustos al ruido gaussiano | Sensibles a ruido no gaussiano y canales no lineales |
| LSTM/GRU (memoria temporal) | Captura estructura temporal larga | Modelos grandes, riesgo de sobreajuste |
| Transformers | Atención global, estado del arte en NLP y visión | Alto coste computacional, necesita muchos datos |
| Autoencoders | Útil con pocas etiquetas, preentrenamiento no supervisado | El espacio latente puede no capturar lo relevante para clasificación |
| Aprendizaje contrastivo | Invariante al SNR, muy prometedor | Diseño de pares de entrenamiento delicado |

## Estructura

```
Modelos_Bajo_SNR/
├── common_snr/      # Features nuevas (HOS, STFT) y sus modelos — NO se llama "common"
│                     #   a propósito, para no chocar con Modelos iniciales/common/ cuando
│                     #   ambos quedan en sys.path a la vez (ver más abajo)
├── hos/             # Entrena HOSClassifier sobre momentos/cumulantes de orden superior
├── stft/            # Entrena STFTCNN sobre el espectrograma STFT
├── lstm_snr/        # Entrena CNNLSTMHybrid (CNN + LSTM) sobre I/Q crudo
└── comparacion/     # No entrena nada: compara los 3 de arriba + baselines de Modelos iniciales
```

`common_snr/` **no duplica** carga de datos ni el loop de entrenamiento: cada notebook hace

```python
import sys
sys.path.insert(0, "../../Modelos iniciales")  # common/: data.py, train.py
sys.path.insert(0, "..")                       # common_snr/: features.py, models.py

from common.data import load_radioml_digital, split_dataset, make_loader
from common.train import train_model, evaluate_by_snr, overall_accuracy, compute_cm
from common_snr.features import compute_hos_features  # o compute_stft
from common_snr.models import HOSClassifier            # o STFTCNN, CNNLSTMHybrid
```

y solo agrega lo específico de la estrategia: la transformación de features (`common_snr/features.py`)
y la arquitectura (`common_snr/models.py`). El dataset, el split (60/20/20, misma semilla), el
loop de entrenamiento (`EarlyStopping`, `ReduceLROnPlateau` opcional) y las utilidades de
evaluación son exactamente los mismos que usa `Modelos iniciales/`.

## Cada notebook de entrenamiento

Misma estructura que en `Modelos iniciales/`: imports/config → carga y transformación de
datos → entrenamiento → evaluación → gráficos. Con dos agregados específicos de esta carpeta:

- La evaluación reporta aparte la accuracy para `SNR <= -10 dB` vs `SNR > -10 dB` (el umbral
  aproximado del límite de Fisher mencionado arriba).
- El gráfico de accuracy vs SNR marca ese umbral con una línea de referencia.

`MODEL_KWARGS` funciona igual que en `Modelos iniciales/`: se edita esa celda para probar otra
profundidad/ancho, y la config queda guardada en `history.pkl` junto con el historial de
entrenamiento para que `comparacion.ipynb` reconstruya el modelo exacto antes de cargar pesos.

## `comparacion/comparacion.ipynb`

Carga los 3 modelos entrenados acá **y** 3 baselines ya entrenados en `../Modelos iniciales/`
(`cnn/`, `tcn/`, `cnn_preproc/` — `ULCNN`), para responder la pregunta real: ¿alguna de estas
estrategias mejora la accuracy específicamente a muy bajo SNR, o el límite de Fisher termina
dominando igual sin importar la representación/arquitectura?

## Agregar una estrategia nueva

Mismo criterio que en `Modelos iniciales/`:

1. La arquitectura va en `common_snr/models.py`.
2. La transformación de features (si aplica) va en `common_snr/features.py`.
3. El notebook de entrenamiento sigue la misma estructura que los existentes (usarlos como
   plantilla) y define su propio `MODEL_KWARGS`.
4. Se agrega su entrada en `MODEL_SPECS` dentro de `comparacion/comparacion.ipynb`.

Nunca copiar código de carga de datos o el loop de entrenamiento dentro de un notebook —
siempre importar desde `../Modelos iniciales/common/` (datos/entrenamiento genéricos) o
`common_snr/` (features/modelos específicos de bajo SNR).
