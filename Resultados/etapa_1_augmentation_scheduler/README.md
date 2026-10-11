# Etapa 1 — augmentation + scheduler (sin curriculum)

Igual que `etapa_0_baseline/`, esta etapa sale de `sweep.py`, no de `snapshot.py` ni de los
notebooks de `Modelos iniciales/`. `metricas.json` se armó a partir de las carpetas de
`Resultados/sweep/` listadas en el campo `fuente` de cada modelo.

## Comando

```bash
python Resultados/sweep.py --archs cnn lstm gru tcn ulcnn --params base --lrs 1e-3 3e-4 \
    --batch-sizes 256 --augment 1 --curriculum 0 --scheduler 1 --epochs 200 --patience 20
```

Esta etapa usa las 5 corridas con **lr 1e-3** (las de 3e-4 quedan en `Resultados/sweep/README.md`,
el análisis del barrido de learning rate). Receta: lr 1e-3, batch 256, augmentation activada,
scheduler activado, **sin** curriculum, 200 épocas tope, paciencia 20, semilla 42.

## Resultados

| Arquitectura | Params | Accuracy global | Accuracy SNR ≤ -10 dB | Épocas |
|---|---|---|---|---|
| CNN (I/Q) | 1.063.048 | 0.521 | 0.143 | 23 |
| LSTM | 208.588 | 0.675 | 0.151 | 64 |
| GRU | 40.428 | 0.677 | 0.157 | 95 |
| TCN (I/Q) | 75.592 | 0.645 | 0.159 | 92 |
| ULCNN | 9.364 | 0.647 | 0.167 | 67 |

## Comparación con la etapa 0 (sin augmentation ni scheduler)

| Arquitectura | Etapa 0 | Etapa 1 | Diferencia |
|---|---|---|---|
| CNN (I/Q) | 0.490 | 0.521 | +0.031 |
| LSTM | 0.179 (colapsó) | 0.675 | +0.496 |
| GRU | 0.603 | 0.677 | +0.074 |
| TCN (I/Q) | 0.625 | 0.645 | +0.020 |
| ULCNN | 0.584 | 0.647 | +0.063 |

La augmentation y el scheduler mejoran a las cinco arquitecturas. El caso más marcado es la
LSTM: con scheduler, el lr efectivo baja cuando la val loss se estanca, y eso es lo que evita
el colapso que se documentó en `etapa_0_baseline/README.md`.
