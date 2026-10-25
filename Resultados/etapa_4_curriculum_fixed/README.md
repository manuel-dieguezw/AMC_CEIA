# Etapa 4 — curriculum por SNR con corte fijo (`--curriculum-mode fixed`)

Igual que las etapas 0-2, sale de `sweep.py`, no de `snapshot.py` ni de los notebooks.
`metricas.json` se armó a partir de las carpetas de `Resultados/sweep/` listadas en `fuente`.

Responde al punto 2 pendiente de "Correcciones del tutor": el criterio de corte
del curriculum original (`--curriculum-mode acc`, el único que existía en la etapa 2) decidía
cuándo pasar de etapa o cortar mirando la **val accuracy sobre todos los SNR**, aunque el modelo
recién estuviera entrenando con un subconjunto de SNR alto — una señal que no mide lo que se
necesitaba medir, y que colapsó al TCN en la etapa 2. Esta etapa prueba la alternativa más simple
que proponía el tutor: **épocas fijas por grupo de SNR, sin mirar accuracy para decidir nada**
(`--curriculum-mode fixed`).

## Comando

```bash
python Resultados/sweep.py --archs cnn lstm gru tcn ulcnn --params base --repr iq \
    --curriculum 1 --curriculum-mode fixed --curriculum-epochs-per-stage 20 \
    --scheduler 1 --augment 1 --epochs 150
```

Receta: la de la etapa 1 (lr por arquitectura, batch 256, augmentation, scheduler) más
curriculum por SNR en modo `fixed` — 5 etapas (grupos de SNR de a 4 niveles, de alto a bajo,
igual que siempre), **20 épocas fijas por etapa** (100 épocas en total, sin early stopping
dentro del curriculum). `cnn`, `tcn` y `ulcnn` se corrieron con `--device dml` (DirectML, GPU
AMD); `lstm` y `gru` en CPU (`nn.LSTM`/`nn.GRU` no
corren en DirectML, bug conocido documentado ahí mismo).

## Resultados

| Arquitectura | Params | Accuracy global | Accuracy SNR ≤ -10 dB | Épocas | Minutos | Device |
|---|---|---|---|---|---|---|
| CNN (I/Q) | 1.063.048 | 0.547 | 0.138 | 100 | 19.5 | dml |
| LSTM | 208.588 | 0.677 | 0.163 | 100 | 256.0 | cpu |
| GRU | 40.428 | 0.673 | 0.152 | 100 | 195.2 | cpu |
| TCN (I/Q) | 75.592 | 0.632 | 0.138 | 100 | 39.0 | dml |
| ULCNN | 9.364 | 0.632 | 0.164 | 100 | 64.1 | dml |

## Comparación con la etapa 1 (sin curriculum) y la etapa 2 (curriculum `acc`, criterio viejo)

| Arquitectura | Etapa 1 (sin curriculum) | Etapa 2 (`acc`, viejo) | Etapa 4 (`fixed`) | Etapa 4 vs Etapa 1 |
|---|---|---|---|---|
| CNN (I/Q) | 0.521 | 0.484 | **0.547** | +0.026 |
| LSTM | 0.675 | 0.673 | 0.677 | +0.003 |
| GRU | 0.677 | 0.600 | 0.673 | −0.004 |
| TCN (I/Q) | 0.645 | **0.213 (colapso)** | 0.632 | −0.013 |
| ULCNN | 0.647 | 0.600 | 0.632 | −0.016 |

## El corte fijo evita el colapso; el curriculum sigue sin aportar una mejora clara

El cambio de criterio de corte **resuelve el problema de la etapa 2**: el TCN ya no colapsa
(0.632 vs el 0.213 que daba con el criterio `acc`), y ninguna de las cinco arquitecturas queda
muy por debajo de su resultado sin curriculum (la mayor caída es ULCNN, −0.016). CNN incluso
mejora un poco (+0.026).

Durante el entrenamiento de CNN en DirectML se observaron picos grandes de `Val Loss` (saltos a
>100 entre épocas, más marcados en las primeras etapas del curriculum, con SNR alto) que no se
habían visto en corridas sin curriculum. A pesar de eso, el checkpoint final (elegido por mejor
val accuracy) terminó en un resultado sano y coherente con el resto — no hubo degradación en el
resultado reportado. Queda sin confirmar si esos picos son un artefacto de DirectML o algo propio
del modo `fixed` (candidato a revisar si se repite el patrón en futuras corridas); no bloqueó
este resultado.

**A SNR ≤ -10 dB no cambia nada relevante** — las tres etapas (1, 2 y 4) están todas pegadas al
azar (~0.14-0.16) en las cinco arquitecturas, consistente con lo que se viene observando en todo
el proyecto: a esos niveles de ruido ningún modelo de esta familia distingue la modulación.

**Conclusión para el punto 2 del tutor:** el criterio de corte del curriculum original (`acc`)
era efectivamente el problema, no el curriculum en sí — con un corte fijo por épocas, el
curriculum deja de romper nada. Dicho esto, tampoco se ve un beneficio claro y consistente sobre
entrenar con todos los SNR desde el inicio (sube un poco en CNN, baja un poco en el resto, dentro
de ±0.03). Falta todavía probar la otra alternativa que proponía el tutor
(`--curriculum-mode stage_val`, validar cada etapa solo con el rango de SNR ya visto) antes de
cerrar la conclusión del punto 2 — ver etapa 5.
