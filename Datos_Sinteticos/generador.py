"""Generador de señales sintéticas para AMC (mismas 8 modulaciones digitales que RadioML 2016.10a).

Genera muestras I/Q de 128 puntos, formato (N, 2, 128) float32, compatibles con
`common.data` (mismo orden alfabético de clases que el LabelEncoder). Pensado para
generar datos "infinitos" on-the-fly; por ahora NO está conectado a los notebooks.

Cadena: bits/símbolos -> conformado de pulso (RRC, o fase continua para FSK) ->
normalizado a potencia 1 -> impairments opcionales (offset de fase/frecuencia) -> AWGN a SNR dado.
"""
import numpy as np
from scipy.signal import fftconvolve

MODULACIONES = ["8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK"]  # orden del LabelEncoder
SNRS = list(range(-20, 20, 2))  # -20..18 dB, como RadioML

N_MUESTRAS = 128
SPS = 8  # muestras por símbolo (RadioML 2016.10a)


# ---------------------------------------------------------------- constelaciones
def _constelacion(mod):
    if mod == "BPSK":
        return np.array([-1, 1], dtype=complex)
    if mod == "QPSK":
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    if mod == "8PSK":
        return np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    if mod == "PAM4":
        return np.array([-3, -1, 1, 3], dtype=complex)
    if mod in ("QAM16", "QAM64"):
        m = int(np.sqrt(int(mod[3:])))
        niveles = np.arange(-(m - 1), m, 2)
        return (niveles[:, None] + 1j * niveles[None, :]).ravel()
    raise ValueError(mod)


def rrc(beta=0.35, sps=SPS, span=11):
    """Filtro raíz de coseno realzado (energía unitaria)."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.zeros_like(n)
    for i, t in enumerate(n):
        if np.isclose(t, 0):
            h[i] = 1 - beta + 4 * beta / np.pi
        elif np.isclose(abs(t), 1 / (4 * beta)):
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))) \
                   / (np.pi * t * (1 - (4 * beta * t) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


# ---------------------------------------------------------------- moduladores (batch)
def _modular_lineal(mod, n, rng, beta=0.35):
    """PSK/QAM/PAM: (n, N_MUESTRAS) complejo, potencia 1."""
    const = _constelacion(mod)
    h = rrc(beta)
    largo = N_MUESTRAS + len(h) + 2 * SPS  # margen para descartar transitorios
    nsim = largo // SPS + 2
    sim = const[rng.integers(0, len(const), size=(n, nsim))]
    up = np.zeros((n, nsim * SPS), dtype=complex)
    up[:, ::SPS] = sim
    y = fftconvolve(up, h[None, :], axes=1)
    off = rng.integers(len(h), len(h) + 2 * SPS, size=n)  # ventana aleatoria (fase de símbolo aleatoria)
    return np.stack([y[i, o:o + N_MUESTRAS] for i, o in enumerate(off)])


def _modular_fsk(mod, n, rng, h_mod=0.5, bt=0.35):
    """CPFSK (pulso rectangular) o GFSK (pulso gaussiano), binaria, fase continua."""
    nsim = N_MUESTRAS // SPS + 6
    bits = rng.choice([-1.0, 1.0], size=(n, nsim))
    up = np.repeat(bits, SPS, axis=1)  # pulso rectangular de duración 1 símbolo
    if mod == "GFSK":
        up = fftconvolve(up, _gauss_pulso(bt)[None, :], axes=1, mode="same")
    fase = np.pi * h_mod * np.cumsum(up, axis=1) / SPS
    y = np.exp(1j * fase)
    off = rng.integers(2 * SPS, 4 * SPS, size=n)
    return np.stack([y[i, o:o + N_MUESTRAS] for i, o in enumerate(off)])


def _gauss_pulso(bt, span=4):
    t = np.arange(-span * SPS // 2, span * SPS // 2 + 1) / SPS
    g = np.exp(-2 * (np.pi * bt * t) ** 2 / np.log(2))
    return g / g.sum()


def _senal_limpia(mod, n, rng):
    if mod in ("CPFSK", "GFSK"):
        x = _modular_fsk(mod, n, rng)
    else:
        x = _modular_lineal(mod, n, rng)
    return x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))  # potencia 1 por muestra


# ---------------------------------------------------------------- canal
def aplicar_canal(x, snr_db, rng, fase_aleatoria=True, max_offset_freq=0.0, escala=1.0):
    """x: (n, L) complejo con potencia 1. snr_db: escalar o (n,).
    max_offset_freq: offset de portadora máximo, en fracción de fs (0 = sin offset)."""
    n, L = x.shape
    if fase_aleatoria:
        x = x * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n, 1)))
    if max_offset_freq > 0:
        df = rng.uniform(-max_offset_freq, max_offset_freq, size=(n, 1))
        x = x * np.exp(2j * np.pi * df * np.arange(L)[None, :])
    snr = np.broadcast_to(np.asarray(snr_db, dtype=float), (n,))[:, None]
    sigma = np.sqrt(10 ** (-snr / 10) / 2)  # potencia señal = 1
    ruido = sigma * (rng.standard_normal((n, L)) + 1j * rng.standard_normal((n, L)))
    return (x + ruido) * escala


# ---------------------------------------------------------------- API
def generar_lote(n, modulaciones=MODULACIONES, snrs=SNRS, rng=None, escala=1.0, **canal):
    """Lote aleatorio balanceado en (modulación, SNR).
    Devuelve X (n, 2, 128) float32, y (n,) int64 [índice en `MODULACIONES`], snr (n,) int64.
    `escala`: RadioML tiene amplitud ~0.006; usar 1.0 (default) o ~0.0075 para imitarla."""
    rng = np.random.default_rng(rng)
    mods = np.asarray(modulaciones)
    y_mod = rng.integers(0, len(mods), size=n)
    snr = rng.choice(np.asarray(snrs), size=n)
    X = np.empty((n, 2, N_MUESTRAS), dtype=np.float32)
    for k, m in enumerate(mods):
        idx = np.where(y_mod == k)[0]
        if len(idx) == 0:
            continue
        z = aplicar_canal(_senal_limpia(m, len(idx), rng), snr[idx], rng, escala=escala, **canal)
        X[idx, 0], X[idx, 1] = z.real, z.imag
    y = np.array([MODULACIONES.index(m) for m in mods[y_mod]], dtype=np.int64)
    return X, y, snr.astype(np.int64)


def generador_infinito(batch_size=256, semilla=None, **kw):
    """Iterador infinito de lotes (X, y, snr); un lote nuevo por iteración (on-the-fly)."""
    rng = np.random.default_rng(semilla)
    while True:
        yield generar_lote(batch_size, rng=rng, **kw)


VERSION = 1  # subir al cambiar la cadena de generación/canal (invalida comparaciones con versiones viejas)


def config(semilla=None, escala=1.0, snrs=SNRS, modulaciones=MODULACIONES, fase_aleatoria=True,
           max_offset_freq=0.0):
    """Dict serializable con todo lo necesario para reproducir/identificar los datos sintéticos.
    Pasarle los mismos argumentos que se usaron en `generar_lote`; se guarda en config.json."""
    return {"data_source": "synthetic", "generador_version": VERSION, "semilla": semilla, "escala": escala,
            "snrs": list(snrs), "modulaciones": list(modulaciones), "n_muestras": N_MUESTRAS, "sps": SPS,
            "rrc_beta": 0.35, "gfsk_bt": 0.35, "fsk_h": 0.5, "fase_aleatoria": fase_aleatoria,
            "max_offset_freq": max_offset_freq}
