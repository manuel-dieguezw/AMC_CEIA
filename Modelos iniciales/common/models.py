"""Arquitecturas de modelos para AMC (RadioML 2016.10a).

Una sola versión de cada arquitectura (con BatchNorm), compartida por los
notebooks cnn/, lstm/, gru/, tcn/ y por comparacion/ — evita que cada notebook
tenga su propia copia de las clases, que terminan divergiendo entre sí.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """CNN de O'Shea et al. (2016) con BatchNorm.

    `channels` es una tupla con un canal de salida por capa convolucional —
    su longitud define la profundidad (defaults: 2 capas de 64, igual a la
    arquitectura original, así los pesos ya entrenados siguen cargando).
    """

    def __init__(self, n_classes=8, channels=(64, 64), kernel_size=3, fc_hidden=128, dropout=0.5):
        super().__init__()
        conv_layers = []
        in_channels = 2 #I y Q
        for out_channels in channels:
            conv_layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 128, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMClassifier(nn.Module):
    """LSTM para clasificación de modulaciones.

    `num_layers` define la profundidad de la LSTM y `hidden_size` su ancho.

    `input_norm=True` agrega un `BatchNorm1d` sobre los canales I/Q antes de la recurrencia.
    La señal de RadioML tiene amplitud minúscula (std ≈ 0.006) y, a diferencia de CNN/TCN (que
    tienen BatchNorm tras la primera conv), la LSTM la recibe cruda: casi no hay señal para las
    compuertas y el entrenamiento se queda en el azar mucho tiempo. Default `False` para que los
    pesos entrenados sin normalizar sigan cargando.
    """

    def __init__(self, input_size=2, hidden_size=128, num_layers=2, n_classes=8, fc_hidden=64, dropout=0.5,
                 input_norm=False):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_size) if input_norm else None
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM no admite dropout con una sola capa
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRUClassifier(nn.Module):
    """GRU para clasificación de modulaciones. Más rápida y estable que LSTM.

    `num_layers` define la profundidad de la GRU y `hidden_size` su ancho.

    `input_norm=True` agrega un `BatchNorm1d` sobre los canales I/Q antes de la recurrencia
    (ver `LSTMClassifier`: la señal cruda es demasiado chica para que la GRU arranque).
    Default `False` para que los pesos entrenados sin normalizar sigan cargando.
    """

    def __init__(self, input_size=2, hidden_size=128, num_layers=2, n_classes=8, fc_hidden=64, dropout=0.5,
                 input_norm=False):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_size) if input_norm else None
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.GRU no admite dropout con una sola capa
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = x.permute(0, 2, 1)  # (batch, 2, 128) -> (batch, 128, 2)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Bloque residual con convolución dilatada — unidad básica de la TCN."""

    def __init__(self, channels, dilation, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.conv(x)  # conexión residual


class TCN(nn.Module):
    """Temporal Convolutional Network para clasificación de modulaciones.

    `dilations` tiene un valor por bloque residual — su longitud define la
    profundidad (cada bloque dobla el campo receptivo) y `channels` el ancho.
    `input_norm=True` agrega un `BatchNorm1d` sobre los canales de entrada antes de la
    proyección (la señal cruda tiene amplitud ≈ 0.005; default False = arquitectura original).
    """

    def __init__(self, n_classes=8, channels=64, dilations=(1, 2, 4, 8), dropout=0.2,
                 input_norm=False):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(2) if input_norm else None
        self.input_proj = nn.Conv1d(2, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels, d, dropout=dropout) for d in dilations])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, n_classes),
        )

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualTCN(nn.Module):
    """TCN de dos ramas: una sobre I/Q y otra sobre Amplitud/Fase de la MISMA señal, con una
    cabeza de clasificación común sobre los dos embeddings concatenados.

    Recibe I/Q crudo `(N, 2, L)`; A/φ se calcula adentro del `forward` (amplitud = sqrt(I²+Q²),
    fase = atan2(Q, I), igual que `data.to_amplitude_phase`). Así se puede usar con los mismos
    loaders, augmentation (`augment_rotate_iq`, que actúa sobre I/Q) y curriculum que el resto, y
    al desplegar solo hace falta I/Q.

    Cada rama tiene su propio `BatchNorm1d(2)` de entrada (I/Q ≈ 0.005, amplitud ≈ 0.005, fase en
    radianes: escalas muy distintas) y su propia pila de `ResidualBlock`. `channels` es el ancho
    de CADA rama: los parámetros son ~2x los de una `TCN` con el mismo `channels` (con
    `channels=40` y 4 bloques queda cerca de los ~75-100k de `tcn/` y `tcn_ap/`).
    """

    def __init__(self, n_classes=8, channels=40, dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.iq_branch = self._make_branch(channels, dilations, dropout)
        self.ap_branch = self._make_branch(channels, dilations, dropout)
        self.classifier = nn.Linear(2 * channels, n_classes)

    @staticmethod
    def _make_branch(channels, dilations, dropout):
        return nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(2, channels, kernel_size=1),
            *[ResidualBlock(channels, d, dropout=dropout) for d in dilations],
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        i, q = x[:, 0], x[:, 1]
        ap = torch.stack([torch.sqrt(i ** 2 + q ** 2), torch.atan2(q, i)], dim=1)
        return self.classifier(torch.cat([self.iq_branch(x), self.ap_branch(ap)], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ComplexConv1d(nn.Module):
    """Convolución 1D compleja (Trabelsi et al., 2018), implementada con dos Conv1d reales
    en vez de tensores de dtype complejo — así el modelo entero sigue siendo float32, más
    simple de cuantizar/exportar a un embebido.

    Recibe la parte real e imaginaria por separado y aplica (A+iB)(a+ib) = (Aa−Bb) + i(Ab+Ba),
    donde A y B son los pesos de dos Conv1d reales independientes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same"):
        super().__init__()
        pad = kernel_size // 2 if padding == "same" else 0
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)

    def forward(self, x_r, x_i):
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_i(x_r) + self.conv_r(x_i)
        return out_r, out_i


class IQCF(nn.Module):
    """I/Q Complex Feature extraction — primer bloque de ULCNN.

    ComplexConv1D → BatchNorm (por separado en parte real e imaginaria — versión "naive" de
    complex batch norm, sin covarianza cruzada real/imaginaria) → ReLU (por parte) → concatenar
    real e imaginaria en el eje de canales. Entrada (N, 2, L) con canal 0 = I, canal 1 = Q;
    salida (N, 2*out_channels, L).
    """

    def __init__(self, out_channels=16, kernel_size=5):
        super().__init__()
        self.complex_conv = ComplexConv1d(1, out_channels, kernel_size, stride=1, padding="same")
        self.bn_r = nn.BatchNorm1d(out_channels)
        self.bn_i = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x_r = x[:, 0:1, :]
        x_i = x[:, 1:2, :]
        out_r, out_i = self.complex_conv(x_r, x_i)
        out_r = F.relu(self.bn_r(out_r))
        out_i = F.relu(self.bn_i(out_i))
        return torch.cat([out_r, out_i], dim=1)


class ChannelShuffle(nn.Module):
    """Mezcla canales entre `groups` grupos (Shuffle-Net) para que la siguiente depthwise
    conv de un FMDR pueda mezclar información entre los sub-grupos de canales."""

    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, length = x.shape
        g = self.groups
        x = x.view(n, g, c // g, length)
        x = x.transpose(1, 2).contiguous()
        return x.view(n, c, length)


class ChannelAttention(nn.Module):
    """Channel attention estilo CBAM: pooling promedio y máximo por canal, pasados por el
    mismo MLP (reducción → expansión), sumados y con sigmoide — reescala cada canal de la
    entrada por su score de atención."""

    def __init__(self, channels, reduction_dim=2):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, channels),
        )

    def forward(self, x):
        avg = x.mean(dim=2)
        mx = x.amax(dim=2)
        attn = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * attn.unsqueeze(-1)


class FMDR(nn.Module):
    """Feature-map Down-sampling & Refinement — bloque repetido `n_fmdr` veces en ULCNN.

    Depthwise conv (stride=2, reduce la longitud temporal a la mitad) → pointwise conv →
    BatchNorm → ReLU → channel shuffle → channel attention.
    """

    def __init__(self, channels=32, kernel_size=5, groups=2, reduction_dim=2):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, stride=2, padding=pad, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.shuffle = ChannelShuffle(groups=groups)
        self.attention = ChannelAttention(channels, reduction_dim=reduction_dim)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.relu(self.bn(x))
        x = self.shuffle(x)
        x = self.attention(x)
        return x


class ULCNN(nn.Module):
    """Ultra-Lightweight CNN para AMC, pensada para despliegue embebido: convolución compleja
    (IQCF) + bloques depthwise-separable con channel shuffle y channel attention (FMDR) +
    fusión multi-escala de features (CLFF) en vez de clasificar solo desde la última capa.

    Entrada I/Q (N, 2, L) — canal 0 = I, canal 1 = Q. `n_cv` (ancho de IQCF) y `n_fmdr`
    (profundidad — cantidad de bloques FMDR) son los principales knobs para probar variantes
    más chicas/grandes. El ancho de los bloques FMDR se deriva como `2 * n_cv` (concatenación
    de parte real e imaginaria en IQCF), no es un parámetro independiente.

    CLFF combina (suma) el global-average-pooling de los últimos 3 bloques FMDR antes de la
    capa densa final — por eso requiere `n_fmdr >= 3`.
    """

    def __init__(self, n_classes=8, n_cv=16, kernel_size=5, n_fmdr=6, shuffle_groups=2, attn_reduction=2):
        super().__init__()
        if n_fmdr < 3:
            raise ValueError("ULCNN combina los últimos 3 bloques FMDR (CLFF); n_fmdr debe ser >= 3")
        pw_channels = 2 * n_cv
        self.iqcf = IQCF(out_channels=n_cv, kernel_size=kernel_size)
        self.fmdr_blocks = nn.ModuleList([
            FMDR(channels=pw_channels, kernel_size=kernel_size, groups=shuffle_groups, reduction_dim=attn_reduction)
            for _ in range(n_fmdr)
        ])
        self.classifier = nn.Linear(pw_channels, n_classes)

    def forward(self, x):
        x = self.iqcf(x)
        pooled = []
        n_blocks = len(self.fmdr_blocks)
        for i, block in enumerate(self.fmdr_blocks):
            x = block(x)
            if i >= n_blocks - 3:  # CLFF: últimos 3 bloques
                pooled.append(x.mean(dim=2))  # global average pooling
        clff = torch.stack(pooled, dim=0).sum(dim=0)
        return self.classifier(clff)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
