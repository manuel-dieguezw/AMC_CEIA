"""Chequeos rápidos + gráfico de constelaciones. Uso: python validar.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generador import MODULACIONES, generar_lote, _senal_limpia, aplicar_canal

rng = np.random.default_rng(0)
X, y, snr = generar_lote(4000, rng=1)
print("forma:", X.shape, X.dtype, "| clases:", np.bincount(y), "| SNRs:", sorted(set(snr))[:3], "...")

# SNR medido vs pedido
for m in ("QPSK", "GFSK"):
    z = _senal_limpia(m, 4000, rng)
    for s in (-10, 0, 10):
        r = aplicar_canal(z, s, rng, fase_aleatoria=False)
        print(m, f"SNR pedido {s:>3} dB -> medido {10 * np.log10(1 / np.mean(np.abs(r - z) ** 2)):.2f} dB")

fig, ax = plt.subplots(2, 8, figsize=(22, 6))
for k, m in enumerate(MODULACIONES):
    z = _senal_limpia(m, 1, rng)
    for f, (s, fila) in enumerate([(18, 0), (0, 1)]):
        r = aplicar_canal(z, s, rng, fase_aleatoria=False)[0]
        ax[fila, k].plot(r.real, r.imag, ".-" if m in ("CPFSK", "GFSK") else ".", ms=3, lw=.3)
        ax[fila, k].set_title(f"{m} @ {s} dB"); ax[fila, k].set_aspect("equal")
plt.tight_layout(); plt.savefig("plano_iq_sintetico.png", dpi=90)
print("guardado plano_iq_sintetico.png")
