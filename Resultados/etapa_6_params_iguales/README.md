# Etapa 6 — comparación a igual cantidad de parámetros (~9.3k)

Igual que las etapas 0-5, sale de `sweep.py`, no de `snapshot.py` ni de los notebooks.
`metricas.json` se armó a partir de las carpetas de `Resultados/sweep/` listadas en `fuente`.

Responde al punto 4 pendiente de "Estado y pendientes" en `CLAUDE.md`: el barrido a igual
cantidad de parámetros (`--params`) estaba disponible en `sweep.py` pero nunca se había corrido
con un `target` distinto de `"base"`. Acá se lleva a las cinco arquitecturas a la capacidad de
`ULCNN` (~9.3-9.4k parámetros, su tamaño natural) para comparar qué tan bien aprovecha cada una
una capacidad chica, en vez de comparar arquitecturas con cantidades de parámetros muy distintas
entre sí (de 9k a 1M, como en las etapas anteriores).

## Comando

`gru` y `lstm` ya tenían una corrida con `--params 9364` de antes (2026-10-19); para completar
`cnn` y `tcn` se corrió:

```bash
python Resultados/sweep.py --archs cnn tcn --params 9364 \
    --augment 1 --scheduler 1 --curriculum 0 --epochs 150 --patience 10
```

Receta: la misma base que la etapa 1 (lr por arquitectura, batch 256, augmentation, scheduler,
sin curriculum), pero con `--params 9364` en vez de `--params base` — `sweep.py` busca para cada
arquitectura la combinación de hiperparámetros (`channels`/`hidden_size`/`n_cv`, etc.) que más se
acerca a esa meta de parámetros. `ULCNN` no necesitó una corrida nueva: su configuración `base`
ya tiene 9.364 parámetros, así que esa meta se definió a partir de ella.

## Resultados

| Arquitectura | Params | Accuracy global | Accuracy SNR ≤ -10 dB | Épocas | Minutos |
|---|---|---|---|---|---|
| CNN | 8.576 | 0.387 | 0.138 | 12 | 1.1 |
| TCN | 9.358 | 0.481 | 0.130 | 23 | 15.3 |
| LSTM | 9.229 | 0.532 | 0.138 | 107 | 73.8 |
| GRU | 9.267 | 0.594 | 0.143 | 75 | 223.9 |
| **ULCNN** | 9.364 | **0.647** | **0.167** | 67 | 53.5 |

## Comparación con la capacidad `base` de cada arquitectura (etapa 1, sin curriculum)

| Arquitectura | Params base (etapa 1) | Acc base | Params ~9.3k (etapa 6) | Acc ~9.3k | Caída |
|---|---|---|---|---|---|
| CNN | 1.063.048 | 0.521 | 8.576 | 0.387 | −0.134 |
| TCN | 75.592 | 0.645 | 9.358 | 0.481 | −0.164 |
| LSTM | 208.588 | 0.675 | 9.229 | 0.532 | −0.143 |
| GRU | 40.428 | 0.677 | 9.267 | 0.594 | −0.083 |
| ULCNN | 9.364 | 0.647 | 9.364 | 0.647 | 0 (ya estaba ahí) |

## A igual capacidad, ULCNN gana claramente

Comprimir las otras cuatro arquitecturas a ~9.3k parámetros (entre 11x y 113x menos que su
tamaño `base`) les cuesta caro: todas caen bastante, y ULCNN —que ya vive naturalmente en esa
capacidad, diseñada específicamente para ser liviana (conv compleja + depthwise/attention,
`FMDR`)— termina siendo la mejor de las cinco a esa cantidad de parámetros, por encima incluso de
GRU (su competidora más cercana en los resultados `base`).

CNN y TCN son las que más sufren la poda: ambas cortan muy rápido por early stopping (12 y 23
épocas, contra 67-107 de GRU/LSTM/ULCNN) y quedan lejos de su propio resultado `base` (CNN
−0.134, TCN −0.164). Esto es consistente con que ambas dependen de varias capas convolucionales
con bastantes canales para funcionar bien — reducir `channels`/`fc_hidden` tan agresivamente las
deja con muy poca capacidad de representación y aparentemente dificulta la optimización (el
entrenamiento corta temprano sin converger). GRU y LSTM, con su mecanismo recurrente, aguantan
mejor la reducción (−0.083 y −0.143) aunque tampoco le ganan a ULCNN.

**A SNR ≤ -10 dB no cambia la conclusión de siempre** — las cinco arquitecturas quedan pegadas al
azar (0.130-0.167), con ULCNN apenas por encima del resto.

**Conclusión para el objetivo del proyecto** ("modelo más liviano que cumpla el objetivo"): a
igual cantidad de parámetros, `ULCNN` no es solo la arquitectura con menos parámetros en su
configuración `base` — es también la que mejor aprovecha una capacidad chica cuando se fuerza a
las demás a bajar a su mismo tamaño. Esto refuerza que el diseño de `ULCNN` (no solo su tamaño)
es lo que la hace la candidata más sólida para despliegue embebido.
