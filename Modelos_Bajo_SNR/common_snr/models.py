"""Arquitecturas para las estrategias de AMC a bajo SNR (ver `../README.md`).

Cada una consume una representación de entrada distinta, calculada en `features.py`:
`HOSClassifier` sobre momentos/cumulantes de orden superior, `STFTCNN`/`SpectrogramULCNN`
sobre el espectrograma (STFT o wavelet), y `CNNLSTMHybrid` sobre I/Q crudo (igual que el resto
de `Modelos iniciales/`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HOSClassifier(nn.Module):
    """MLP sobre momentos/cumulantes de orden superior (`features.compute_hos_features`).

    La robustez a bajo SNR se espera de la *feature* (los cumulantes de orden ≥4 son poco
    sensibles al ruido gaussiano), no de la arquitectura — por eso el clasificador es
    deliberadamente chico. `hidden` es una tupla con el ancho de cada capa oculta; su
    longitud define la profundidad.
    """

    def __init__(self, n_features=16, n_classes=8, hidden=(32, 16), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class STFTCNN(nn.Module):
    """CNN 2D sobre una representación tiempo-frecuencia compleja de 2 canales (parte real e
    imaginaria) — STFT (`features.compute_stft`) o wavelet (`features.compute_cwt`), ambas
    devuelven el mismo formato `(N, 2, F, T)`, así que la arquitectura no necesita distinguirlas.
    `channels` define la profundidad (un valor por capa)."""

    def __init__(self, n_classes=8, channels=(16, 32), kernel_size=3, fc_hidden=64, dropout=0.5):
        super().__init__()
        conv_layers = []
        in_channels = 2  # real + imaginaria (STFT o wavelet)
        for out_channels in channels:
            conv_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # evita depender del tamaño exacto del espectrograma
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNLSTMHybrid(nn.Module):
    """CNN (extracción local) + LSTM (memoria temporal larga) sobre I/Q crudo — versión
    liviana inspirada en MCLDNN. La idea es que a bajo SNR la estructura de la modulación
    queda dispersa en el tiempo, y una LSTM integrando muchas muestras puede recuperarla
    mejor que una CNN puramente local (`common/models.py::CNNBaseline`).
    """

    def __init__(self, n_classes=8, conv_channels=32, kernel_size=5, hidden_size=64,
                 num_layers=1, fc_hidden=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
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
        x = self.conv(x)          # (N, conv_channels, L)
        x = x.permute(0, 2, 1)    # (N, L, conv_channels)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ChannelShuffle2D(nn.Module):
    """Versión 2D de `ULCNN`'s `ChannelShuffle` (`Modelos iniciales/common/models.py`) — mezcla
    canales entre `groups` grupos antes del channel attention."""

    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        g = self.groups
        x = x.view(n, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(n, c, h, w)


class ChannelAttention2D(nn.Module):
    """Versión 2D de `ULCNN`'s `ChannelAttention` — pooling promedio y máximo sobre ambos ejes
    espaciales (frecuencia y tiempo), pasados por el mismo MLP compartido, sumados y con
    sigmoide, para reescalar cada canal."""

    def __init__(self, channels, reduction_dim=2):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, channels),
        )

    def forward(self, x):
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        attn = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class FMDR2D(nn.Module):
    """Versión 2D del bloque `FMDR` de `ULCNN` (`Modelos iniciales/common/models.py`):
    depthwise conv 2D (stride=2, reduce frecuencia y tiempo a la mitad) + pointwise conv +
    BatchNorm + ReLU + channel shuffle + channel attention."""

    def __init__(self, channels=16, kernel_size=3, groups=2, reduction_dim=2):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=pad, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.shuffle = ChannelShuffle2D(groups=groups)
        self.attention = ChannelAttention2D(channels, reduction_dim=reduction_dim)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.relu(self.bn(x))
        x = self.shuffle(x)
        x = self.attention(x)
        return x


class SpectrogramULCNN(nn.Module):
    """"ULCNN 2D": aplica los mismos trucos de liviandad de `ULCNN` (depthwise-separable +
    channel shuffle + channel attention, ver `Modelos iniciales/common/models.py`) sobre una
    representación tiempo-frecuencia (STFT o wavelet, `(N, 2, F, T)`) en vez del CNN 2D
    genérico de `STFTCNN`. La hipótesis es que el channel attention puede aprender a ignorar
    bandas de frecuencia/instantes dominados por ruido, algo que una CNN 2D simple no hace
    explícitamente — justo el problema central a bajo SNR.

    A diferencia de `ULCNN` (que arranca de I/Q, 1 canal complejo, y usa `IQCF` para separar
    parte real/imaginaria vía convolución compleja), acá la entrada ya viene con 2 canales
    reales (real/imaginaria del STFT o wavelet), así que alcanza con una proyección 1x1 antes
    de los bloques `FMDR2D`. CLFF combina el average-pooling de los últimos 3 bloques, igual
    que en `ULCNN` — por eso requiere `n_fmdr >= 3`.
    """

    def __init__(self, n_classes=8, channels=16, kernel_size=3, n_fmdr=4, shuffle_groups=2, attn_reduction=2):
        super().__init__()
        if n_fmdr < 3:
            raise ValueError("SpectrogramULCNN combina los últimos 3 bloques FMDR2D (CLFF); n_fmdr debe ser >= 3")
        self.input_proj = nn.Conv2d(2, channels, kernel_size=1)  # real + imaginaria (STFT o wavelet)
        self.fmdr_blocks = nn.ModuleList([
            FMDR2D(channels=channels, kernel_size=kernel_size, groups=shuffle_groups, reduction_dim=attn_reduction)
            for _ in range(n_fmdr)
        ])
        self.classifier = nn.Linear(channels, n_classes)

    def forward(self, x):
        x = self.input_proj(x)
        pooled = []
        n_blocks = len(self.fmdr_blocks)
        for i, block in enumerate(self.fmdr_blocks):
            x = block(x)
            if i >= n_blocks - 3:  # CLFF: últimos 3 bloques
                pooled.append(x.mean(dim=(2, 3)))  # global average pooling
        clff = torch.stack(pooled, dim=0).sum(dim=0)
        return self.classifier(clff)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FusionClassifier(nn.Module):
    """Combina 3 representaciones de la misma señal (I/Q crudo, HOS, STFT) en un solo modelo:
    cada una pasa por su propia rama liviana hasta un embedding chico, los 3 embeddings se
    concatenan, y una capa final clasifica sobre la fusión.

    La hipótesis: cada representación es robusta a bajo SNR por una razón distinta (HOS ante
    ruido gaussiano, STFT ante estructura dispersa en el tiempo, I/Q retiene toda la
    información sin pérdida) — combinarlas podría compensar los puntos débiles de cada una por
    separado (ver comparación en `../README.md`).

    Recibe un único tensor de entrada `(N, iq_size + hos_dim + stft_size)` — ver
    `features.pack_fusion_features` — y lo desempaqueta internamente según `iq_shape`,
    `hos_dim`, `stft_shape`, para poder seguir usando `common/train_model` y compañía sin
    modificarlos (esperan un solo tensor `X`, no una tupla de 3).
    """

    def __init__(self, n_classes=8, iq_shape=(2, 128), hos_dim=16, stft_shape=(2, 32, 9),
                 embed_dim=16, conv_channels=16, kernel_size=3, dropout=0.3):
        super().__init__()
        self.iq_shape = iq_shape
        self.hos_dim = hos_dim
        self.stft_shape = stft_shape

        self.iq_branch = nn.Sequential(
            nn.Conv1d(iq_shape[0], conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_channels, embed_dim),
        )
        self.hos_branch = nn.Sequential(
            nn.Linear(hos_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.stft_branch = nn.Sequential(
            nn.Conv2d(stft_shape[0], conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv_channels, embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x):
        iq_size = self.iq_shape[0] * self.iq_shape[1]
        stft_size = self.stft_shape[0] * self.stft_shape[1] * self.stft_shape[2]

        x_iq = x[:, :iq_size].reshape(-1, *self.iq_shape)
        x_hos = x[:, iq_size:iq_size + self.hos_dim]
        x_stft = x[:, iq_size + self.hos_dim:iq_size + self.hos_dim + stft_size].reshape(-1, *self.stft_shape)

        emb_iq = self.iq_branch(x_iq)
        emb_hos = self.hos_branch(x_hos)
        emb_stft = self.stft_branch(x_stft)

        fused = torch.cat([emb_iq, emb_hos, emb_stft], dim=1)
        return self.classifier(fused)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
