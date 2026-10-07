# Datos_Sinteticos

Generador de señales I/Q sintéticas (numpy/scipy) para las 8 modulaciones de RadioML 2016.10a:
8PSK, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK. **No está conectado a los notebooks**: es la base
para, más adelante, entrenar con datos generados on-the-fly y usar RadioML solo para test.

```python
from generador import generar_lote, generador_infinito
X, y, snr = generar_lote(1024, rng=0)            # X: (1024, 2, 128) float32
for X, y, snr in generador_infinito(256): ...     # lotes nuevos indefinidamente
```

- `y` usa el mismo orden alfabético que el `LabelEncoder` de `common.data`.
- SNR uniforme en -20..18 dB (paso 2); se puede pasar `snrs=[...]`, `modulaciones=[...]`.
- `escala`: RadioML tiene std ≈ 0.006; el default (1.0) deja potencia de señal = 1. Para imitar
  la escala del dataset, `escala≈0.0075` (los modelos con BatchNorm de entrada no lo necesitan).
- Canal opcional: `fase_aleatoria=True`, `max_offset_freq` (fracción de fs). No modela fading ni
  offset de reloj como RadioML (se generó con GNU Radio); se puede agregar en `aplicar_canal`.
- `python validar.py` chequea SNR medido vs pedido y guarda `plano_iq_sintetico.png`.
