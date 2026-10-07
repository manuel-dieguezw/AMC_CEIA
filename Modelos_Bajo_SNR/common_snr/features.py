"""Extracción de características para AMC a bajo SNR.

Se llama `common_snr` (no `common`) a propósito: los notebooks de esta carpeta también
importan `common.data`/`common.train` de `../../Modelos iniciales/common/` (para no duplicar
carga de datos ni el loop de entrenamiento), y ambos quedan en `sys.path` al mismo tiempo —
si este módulo también se llamara `common` colisionaría con ese import.
"""
import numpy as np
import pywt
from scipy.signal import stft


def compute_hos_features(X):
    """Momentos y cumulantes de orden superior (HOS) de la señal I/Q, por muestra.

    X: (N, 2, L) I/Q → devuelve (N, 16) con partes real/imaginaria de un subset de momentos
    M_pq = E[z^(p-q) · conj(z)^q] (con z = I + jQ, normalizada a potencia unitaria) y los
    cumulantes C40, C42, C63 (Swami & Sadler, 1998) derivados de ellos — el set más citado en
    la literatura de AMC por ser robusto a bajo SNR: al ser de orden ≥4, el ruido gaussiano
    (cuyos cumulantes de orden >2 son ~0) los contamina menos que a los momentos de orden 2.
    """
    I, Q = X[:, 0, :], X[:, 1, :]
    z = (I + 1j * Q).astype(np.complex64)

    # Normalizar a potencia unitaria: sin esto, la varianza del ruido (que crece si el SNR baja
    # y la señal se mantiene a potencia fija) desplazaría los momentos de forma espuria.
    power = np.mean(np.abs(z) ** 2, axis=1, keepdims=True)
    z = z / np.sqrt(power + 1e-12)

    def moment(p, q):
        return np.mean(z ** (p - q) * np.conj(z) ** q, axis=1)

    M20, M21, M22 = moment(2, 0), moment(2, 1), moment(2, 2)
    M40, M41, M42, M43 = moment(4, 0), moment(4, 1), moment(4, 2), moment(4, 3)
    M60, M63 = moment(6, 0), moment(6, 3)

    C40 = M40 - 3 * M20 ** 2
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2
    C63 = M63 - 9 * M21 * M42 + 12 * M21 ** 3 - 3 * M20 * M43 - 3 * M22 * M41 + 18 * M20 * M21 * M22

    feats = np.stack(
        [
            M20.real, M20.imag, M21.real,
            M40.real, M40.imag, M41.real, M42.real, M43.real,
            M60.real, M63.real, M63.imag,
            C40.real, C40.imag, C42.real, C63.real, C63.imag,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def compute_stft(X, nperseg=32, noverlap=16):
    """Espectrograma STFT complejo de la señal I/Q, por muestra.

    X: (N, 2, L) → z = I + jQ → STFT (bilateral, `axis=-1` de scipy.signal.stft procesa cada
    fila del batch de forma independiente) → devuelve (N, 2, F, T) con parte real e imaginaria
    del STFT como 2 canales, listo para una CNN 2D.
    """
    I, Q = X[:, 0, :], X[:, 1, :]
    z = (I + 1j * Q).astype(np.complex64)
    _, _, Zxx = stft(z, nperseg=nperseg, noverlap=noverlap, axis=-1, return_onesided=False)
    out = np.stack([Zxx.real, Zxx.imag], axis=1)
    return out.astype(np.float32)


def compute_cwt(X, scales=np.arange(1, 33), wavelet="cmor1.5-1.0"):
    """Transformada wavelet continua (CWT) compleja de la señal I/Q, por muestra.

    X: (N, 2, L) → z = I + jQ → CWT con una wavelet de Morlet compleja (`cmor`) en `scales`
    escalas → devuelve (N, 2, len(scales), L) con parte real e imaginaria como 2 canales,
    listo para una CNN 2D — mismo formato que `compute_stft`.

    A diferencia del STFT (resolución tiempo-frecuencia fija, limitada por incertidumbre de
    Heisenberg — ver `../README.md`), la CWT da resolución multi-escala: mejor resolución
    temporal en escalas finas (~frecuencias altas) y mejor resolución frecuencial en escalas
    gruesas (~frecuencias bajas).

    `pywt.cwt` no está vectorizado sobre un batch, así que se itera muestra por muestra — para
    L=128 y 32 escalas tarda ~1ms/muestra (~3 min para las 160k señales del dataset completo).
    """
    I, Q = X[:, 0, :], X[:, 1, :]
    z = (I + 1j * Q).astype(np.complex64)
    n_samples, length = z.shape
    n_scales = len(scales)

    out = np.empty((n_samples, 2, n_scales, length), dtype=np.float32)
    for i in range(n_samples):
        coeffs, _ = pywt.cwt(z[i], scales, wavelet)
        out[i, 0] = coeffs.real
        out[i, 1] = coeffs.imag
    return out


def pack_fusion_features(X_iq, X_hos, X_stft):
    """Empaqueta 3 representaciones de la misma señal (I/Q, HOS, STFT) en un solo vector
    plano por muestra, concatenando cada una aplanada.

    Con esto, `common/make_loader` y `common/train_model` (pensados para un único tensor `X`)
    funcionan sin cambios — `models.FusionClassifier.forward` desempaqueta el vector de vuelta
    a sus 3 partes con las shapes que le pasás por `MODEL_KWARGS` (`iq_shape`, `hos_dim`,
    `stft_shape`), así que estas 3 llamadas deben usar la misma config con la que se entrenó.
    """
    n = X_iq.shape[0]
    return np.concatenate(
        [X_iq.reshape(n, -1), X_hos.reshape(n, -1), X_stft.reshape(n, -1)], axis=1
    ).astype(np.float32)
