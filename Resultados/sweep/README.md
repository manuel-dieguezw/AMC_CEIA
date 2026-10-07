# Barrido de learning rate (receta común, 2026-10-04 a 2026-10-06)

Este barrido no es una etapa de `snapshot.py`: no copia modelos de los notebooks, sino que entrena
configuraciones nuevas con `sweep.py`. Sus resultados están en esta carpeta, junto a `resumen.csv`.

## Comando

```bash
python Resultados/sweep.py --archs cnn lstm gru tcn ulcnn --params base --lrs 1e-3 3e-4 \
    --batch-sizes 256 --augment 1 --curriculum 0 --scheduler 1 --epochs 200 --patience 20
```

Receta: batch 256, augmentation activada, curriculum desactivado, `ReduceLROnPlateau` activado,
épocas tope 200, paciencia 20, semilla 42. Se barrió el lr en {1e-3, 3e-4}.

## Qué hay en esta carpeta

- Diez corridas nuevas, con nombre `<arch>_iq_p<params>_lr<lr>_bs256_aug1_cur0_sch_seed42`.
- Las corridas anteriores (notebooks y barridos previos) también aparecen en `resumen.csv`.
  Para este análisis, filtrar: `curriculum = False`, `batch_size = 256` y `_sch` en el nombre.
- `resumen.csv` y `accuracy_vs_parametros.png` se regeneran con `python Resultados/sweep.py --resumen`.

## Resultados (una semilla)

| Arquitectura | acc lr 1e-3 | acc lr 3e-4 | Épocas (lr 3e-4) | Minutos (lr 3e-4) |
|---|---|---|---|---|
| CNN I/Q | 0.521 | 0.543 | 22 | 16.7 |
| LSTM | 0.675 | 0.680 | 68 | 370.8 |
| GRU | 0.677 | 0.677 | 143 | 628.6 |
| TCN | 0.645 | 0.643 | 93 | 516.4 |
| ULCNN | 0.647 | 0.650 | 174 | 134.3 |

El lr casi no afecta el resultado (diferencias ≤ 0.5 pp salvo la CNN, que además tuvo
entrenamiento inestable con lr 1e-3). Con una sola semilla no se puede separar ruido de efecto.

## Por qué tardó tanto

Tiempo total de las diez corridas nuevas: ~3.000 min (unas 50 horas, en CPU).

1. **Costo por época según la arquitectura.** Con augmentation el train tiene 384.000 muestras.
   LSTM, GRU y TCN tardan ~5 min por época; CNN y ULCNN, ~1 min. Las recurrentes procesan 128
   pasos en secuencia y en CPU no se paralelizan.
2. **Paciencia 20 en vez de 10.** Cada corrida que cortó por early stopping gastó al menos 20
   épocas extra sin mejora. Con 10, las corridas de LSTM/GRU/TCN habrían durado la mitad.
3. **La val loss es ruidosa.** Oscila mucho entre épocas (por ejemplo, la CNN a 5–18), y eso
   reinicia el contador de paciencia cada vez que aparece un mínimo nuevo, lo que alarga las corridas.
4. **Ejecución secuencial.** Las diez corridas se hicieron una tras otra en la misma máquina.

## Qué cambiar si el costo es un problema

- Paciencia 10: recorta ~30–40% del tiempo de las recurrentes.
- Entrenar solo con lr 1e-3 (el de los notebooks): evita duplicar cada configuración.
- Reducir `train_subset` para barridos exploratorios y dejar el train completo para la corrida final.
