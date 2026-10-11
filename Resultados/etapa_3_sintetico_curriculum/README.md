# Etapa 3 — datos sintéticos + curriculum (+ scheduler)

Igual que las etapas 0, 1 y 2, sale de `sweep.py`, no de `snapshot.py` ni de los notebooks.
`metricas.json` se armó a partir de las carpetas de `Resultados/sweep/` listadas en `fuente`.

**Train y val son sintéticos** (`Datos_Sinteticos/generador.py`, on-the-fly); **el test es
siempre RadioML real**, para que el número sea comparable con las demás etapas.

## Comando

```bash
python Resultados/sweep.py --archs cnn lstm gru tcn ulcnn --params base --lrs 1e-3 \
    --batch-sizes 256 --curriculum 1 --scheduler 1 --epochs 200 --patience 20 --data-source synthetic
```

Receta: lr 1e-3, batch 256, scheduler activado, curriculum activado (mismos defaults que los
notebooks), train/val sintéticos (384.000 / 32.000 muestras, escala 0.0075), test RadioML real.
`augment` no aplica a datos sintéticos (ver `Resultados/sweep.py`).

## Resultados

| Arquitectura | Params | Accuracy global | Accuracy SNR ≤ -10 dB | Épocas | Minutos |
|---|---|---|---|---|---|
| CNN (I/Q) | 1.063.048 | 0.458 | 0.132 | 29 | 11.6 |
| LSTM | 208.588 | 0.569 | 0.163 | 40 | 103.8 |
| GRU | 40.428 | 0.560 | 0.165 | 39 | 96.8 |
| TCN (I/Q) | 75.592 | 0.489 | 0.138 | 30 | 61.3 |
| ULCNN | 9.364 | 0.556 | 0.154 | 32 | 11.5 |

## Comparación con la etapa 2 (RadioML real + curriculum + scheduler)

| Arquitectura | Etapa 2 (real) | Etapa 3 (sintético) | Diferencia |
|---|---|---|---|
| CNN (I/Q) | 0.484 | 0.458 | −0.026 |
| LSTM | 0.673 | 0.569 | −0.104 |
| GRU | 0.600 | 0.560 | −0.040 |
| TCN (I/Q) | **0.213** (colapsó) | **0.489** | **+0.276** |
| ULCNN | 0.600 | 0.556 | −0.044 |

## Lectura

- **El TCN no colapsa con datos sintéticos** (0.489 vs 0.213 con RadioML real), aunque sigue
  siendo la peor de las cinco en esta etapa. Es un indicio de que el colapso del TCN con
  curriculum + I/Q real tiene que ver con alguna particularidad de los datos reales (quizás el
  fading o el offset de reloj que el generador no modela), no con el curriculum en sí.
- **El resto (CNN, LSTM, GRU, ULCNN) rinde peor con datos sintéticos que con RadioML real**,
  entre 3 y 10 puntos porcentuales. Es esperable: el generador no tiene canal con fading ni
  offset de reloj (ver `Datos_Sinteticos/README.md`), así que el train sintético es "más fácil"
  en algunos aspectos y le falta variabilidad que sí tiene el test real.
- **La val loss es mucho más alta y ruidosa que con RadioML real** (llega a 100+ en CNN y TCN,
  contra 2-20 en la etapa 2). El modelo se vuelve muy confiado y se equivoca fuerte en algunas
  épocas, aunque la val accuracy siga mejorando. No se investigó la causa exacta.
- Ninguna de las cinco arquitecturas llegó a superar a su versión con RadioML real + curriculum,
  así que **por ahora entrenar con datos sintéticos no mejora el resultado**, aunque evita el
  colapso del TCN. Falta la comparación equivalente **sin curriculum** (`--curriculum 0`) para
  aislar si la diferencia viene del curriculum o de la fuente de datos.
