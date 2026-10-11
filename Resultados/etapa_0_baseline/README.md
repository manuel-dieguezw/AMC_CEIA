# Etapa 0 — baseline (sin augmentation, scheduler ni curriculum)

A diferencia de la etapa 0 anterior (ver `Resultados/etapa_0_original/`), esta no viene de
`snapshot.py` ni de los notebooks de `Modelos iniciales/`. Sale de cinco corridas de `sweep.py`,
independientes de los notebooks. Por eso **no hay `.pt`, `history.pkl` ni notebooks copiados
acá** — solo `metricas.json`, armado a partir de las carpetas de `Resultados/sweep/` listadas
en el campo `fuente` de cada modelo.

## Comando

```bash
python Resultados/sweep.py --archs cnn lstm gru tcn ulcnn --params base --lrs 1e-3 \
    --batch-sizes 256 --augment 0 --curriculum 0 --scheduler 0 --epochs 200 --patience 20
```

Receta: lr 1e-3, batch 256, **sin** augmentation, **sin** scheduler, **sin** curriculum,
200 épocas tope, paciencia 20, semilla 42, una corrida por arquitectura con su configuración
base (`model_kwargs` de cada notebook).

## Resultados

| Arquitectura | Params | Accuracy global | Accuracy SNR ≤ -10 dB | Épocas | Minutos |
|---|---|---|---|---|---|
| CNN (I/Q) | 1.063.048 | 0.490 | 0.136 | 27 | 5.1 |
| LSTM | 208.588 | **0.179** | 0.153 | 21 | 27.2 |
| GRU | 40.428 | 0.603 | 0.152 | 52 | 58.0 |
| TCN (I/Q) | 75.592 | 0.625 | 0.155 | 83 | 107.3 |
| ULCNN | 9.364 | 0.584 | 0.165 | 42 | 8.4 |

## La LSTM colapsa con esta receta — a propósito

Con lr 1e-3 y sin scheduler, el train loss de la LSTM se queda clavado en ≈ 2.08 (`ln(8)`,
la entropía de predecir las 8 clases por igual) durante las 21 épocas: el modelo no aprende
nada. Es el mismo comportamiento que ya se había documentado en `lstm/lstm.ipynb` ("con 1e-3
la LSTM es errática") y el que apareció con el curriculum. Se decidió **dejar este resultado
tal cual**, para que la etapa muestre el colapso en vez de ocultarlo con un lr distinto solo
para este modelo.

## Diferencia con `snapshot.py` / `etapa_0_original`

- `snapshot.py` copia los `.pt` y notebooks que vos corriste en Jupyter, y reconstruye el
  modelo desde `history.pkl`. Esta etapa, en cambio, entrenó con `sweep.py`, sin pasar por
  los notebooks — las carpetas de `Modelos iniciales/` no se tocaron.
- Hay una diferencia menor en los parámetros: `sweep.py` reconstruye cada arquitectura con los
  kwargs de `ARCHS` (en el propio script), que no siempre coinciden con el notebook al dígito.
  El TCN de esta etapa tiene 75.592 parámetros contra 75.596 del notebook actual (una diferencia
  de 4, por cómo cuenta `input_norm`). No afecta la comparación, pero vale la aclaración.
