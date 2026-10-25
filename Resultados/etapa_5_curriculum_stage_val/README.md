# Etapa 5 — curriculum por SNR con validación por etapa (`--curriculum-mode stage_val`)

Igual que las etapas 0-2 y 4, sale de `sweep.py`, no de `snapshot.py` ni de los notebooks.
`metricas.json` se armó a partir de las carpetas de `Resultados/sweep/` listadas en `fuente`.

Segunda alternativa del punto 2 pendiente de "Correcciones del tutor" (la primera
fue `fixed`, ver `etapa_4_curriculum_fixed/`). Acá, dentro de cada etapa del curriculum, la val
accuracy que decide si "ya no hay progreso" se mide **solo con las muestras de val en el rango de
SNR ya visto** (no con el val set completo, como hacía el modo `acc` original) — el val set
completo se sigue usando únicamente para elegir el checkpoint final.

## Comando

```bash
python Resultados/sweep.py --archs cnn lstm gru tcn ulcnn --params base --repr iq \
    --curriculum 1 --curriculum-mode stage_val --scheduler 1 --augment 1 --epochs 150
```

Receta: la misma que la etapa 4 (lr por arquitectura, batch 256, augmentation, scheduler), con
`--curriculum-mode stage_val` en vez de `fixed`. `cnn`, `tcn` y `ulcnn` corridos con
`--device dml`; `lstm` y `gru` en CPU.

## Resultados

| Arquitectura | Params | Accuracy global | Accuracy SNR ≤ -10 dB | Épocas | Minutos | Device |
|---|---|---|---|---|---|---|
| CNN (I/Q) | 1.063.048 | 0.417 | 0.112 | 30 | 5.8 | dml |
| LSTM | 208.588 | 0.672 | 0.161 | 51 | 136.2 | cpu |
| GRU | 40.428 | 0.598 | 0.144 | 42 | 68.3 | cpu |
| TCN (I/Q) | 75.592 | 0.605 | 0.144 | 38 | 14.3 | dml |
| ULCNN | 9.364 | 0.614 | 0.161 | 35 | 19.2 | dml |

## Comparación con las etapas 1, 2 y 4

| Arquitectura | E1 (sin curriculum) | E2 (`acc`, viejo) | E4 (`fixed`) | E5 (`stage_val`) |
|---|---|---|---|---|
| CNN (I/Q) | 0.521 | 0.484 | **0.547** | 0.417 |
| LSTM | 0.675 | 0.673 | **0.677** | 0.672 |
| GRU | **0.677** | 0.600 | 0.673 | 0.598 |
| TCN (I/Q) | **0.645** | 0.213 (colapso) | 0.632 | 0.605 |
| ULCNN | **0.647** | 0.600 | 0.632 | 0.614 |

## `stage_val` evita el colapso, pero corta demasiado temprano para ganarle a `fixed`

Como `fixed`, `stage_val` **evita el colapso del TCN** (0.605 vs el 0.213 de la etapa 2). Pero a
diferencia de `fixed`, en 4 de las 5 arquitecturas queda **por debajo tanto del baseline sin
curriculum (etapa 1) como de `fixed` (etapa 4)** — la excepción es LSTM, donde las cuatro etapas
dan prácticamente lo mismo (esa arquitectura parece poco sensible al curriculum en general).

La causa más probable: `stage_val` entrenó muchas menos épocas que `fixed` en las cinco
arquitecturas (30-51 épocas vs las 100 fijas de `fixed`). Arreglar la métrica de validación
(medir solo el rango de SNR visto) corrige la señal, pero `_train_with_curriculum` sigue usando
esa señal para decidir cuándo cortar cada etapa (`patience`/`min_gain`, igual que el modo `acc`
original) — y sigue cortando antes de que el modelo termine de converger, nada más que ahora con
una métrica más razonable. El caso más notorio es CNN: corta a las 30 épocas con un resultado
peor que el baseline sin curriculum.

**A SNR ≤ -10 dB sigue sin cambios relevantes** — las cuatro etapas (1, 2, 4, 5) están todas
pegadas al azar (~0.11-0.16) en las cinco arquitecturas.

**Conclusión para el punto 2 del tutor:** de las dos alternativas probadas, **`fixed` es la que
mejor resuelve el problema** — evita el colapso y entrena las épocas completas, quedando cerca
(o mejor) que el baseline sin curriculum. `stage_val` corrige la métrica de validación pero
hereda el mismo patrón de corte temprano que tenía el modo `acc`, solo que con una señal más
sana — alcanzaría con relajar `patience`/`min_gain` por etapa o fijar más épocas mínimas, pero no
se probó esa variante. **Para el informe: no usar curriculum en la comparación principal**
(ninguno de los tres modos mejora de forma clara y consistente sobre entrenar con todos los SNR
desde el inicio), y si se reporta un modo de curriculum como el "correcto", usar `fixed`
(etapa 4), no `stage_val` ni el `acc` original.
