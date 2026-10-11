# Etapa 2 — curriculum por SNR (+ augmentation + scheduler)

Igual que las etapas 0 y 1, sale de `sweep.py`, no de `snapshot.py` ni de los notebooks.
`metricas.json` se armó a partir de las carpetas de `Resultados/sweep/` listadas en `fuente`.

## Comando

```bash
python Resultados/sweep.py --archs cnn lstm gru tcn ulcnn --params base --lrs 1e-3 \
    --batch-sizes 256 --augment 1 --curriculum 1 --scheduler 1 --epochs 200 --patience 20
```

Receta: la de la etapa 1 (lr 1e-3, batch 256, augmentation, scheduler) más **curriculum por
SNR** activado, con los mismos valores por defecto que usan los notebooks
(`group_size=4`, `patience=3`, `min_gain=0.005`, `max_epochs_per_stage=15`).

## Resultados

| Arquitectura | Params | Accuracy global | Accuracy SNR ≤ -10 dB | Épocas | Minutos |
|---|---|---|---|---|---|
| CNN (I/Q) | 1.063.048 | 0.484 | 0.145 | 32 | 13.7 |
| LSTM | 208.588 | 0.673 | 0.156 | 40 | 100.3 |
| GRU | 40.428 | 0.600 | 0.145 | 37 | 80.3 |
| **TCN (I/Q)** | 75.592 | **0.213** | 0.164 | 22 | 25.9 |
| ULCNN | 9.364 | 0.600 | 0.149 | 26 | 10.9 |

## Comparación con la etapa 1 (misma receta, sin curriculum)

| Arquitectura | Etapa 1 | Etapa 2 (curriculum) | Diferencia |
|---|---|---|---|
| CNN (I/Q) | 0.521 | 0.484 | −0.037 |
| LSTM | 0.675 | 0.673 | −0.002 |
| GRU | 0.677 | 0.600 | −0.077 |
| TCN (I/Q) | 0.645 | **0.213** | **−0.432** |
| ULCNN | 0.647 | 0.600 | −0.047 |

## El curriculum no mejora a ninguna arquitectura, y colapsa al TCN

Con esta receta (augmentation + scheduler), el curriculum **empeora las cinco arquitecturas**.
El caso más marcado es el TCN, que colapsa igual que ya se había visto en corridas anteriores
con esta misma combinación (curriculum + I/Q): el val loss se dispara entre 4 y 10 de una época
a otra, mientras el train loss baja con normalidad — el modelo se sobreajusta a cada grupo de
SNR que entra y no generaliza al agregar el siguiente. El `train_model` corta la primera etapa
en la época 19 sin pasar del azar, y nunca se recupera.

La CNN, la GRU y la ULCNN pierden entre 4 y 8 puntos porcentuales sin colapsar. La LSTM es la
única que queda prácticamente igual.

**Conclusión para el punto 2 del tutor:** con la receta común (augmentation + scheduler), el
curriculum por SNR no ayuda — empeora el resultado en las cinco arquitecturas probadas, y arruina
al TCN. No se recomienda usarlo en la comparación principal del proyecto.
