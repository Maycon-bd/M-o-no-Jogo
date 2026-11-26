"""Rasterização dos pontos e inferência da rede convolucional em NumPy."""

from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from gestos import DEDOS, GESTOS

PESOS_CNN = Path(__file__).resolve().parent / "modelos" / "cnn.npz"
PESOS_CNN_APRIMORADA = PESOS_CNN.with_name("cnn_aprimorada.npz")
PESOS_CNN_JOGO = PESOS_CNN.with_name("cnn_faixa_95.npz")
LADO = 64


def rasterizar(vetor):
    """Desenha os pontos relativos ao pulso, preservando o sentido do gesto."""
    xy = np.asarray(vetor, dtype=np.float32).reshape(21, 3)[:, :2]
    centro = (xy.min(axis=0) + xy.max(axis=0)) / 2
    pixels = np.rint((xy - centro) * 25 + (LADO - 1) / 2).astype(np.int32)
    imagem = np.zeros((LADO, LADO), dtype=np.uint8)
    for dedos in DEDOS:
        cv2.polylines(imagem, [pixels[list(dedos)]], False, 180, 2, cv2.LINE_AA)
    for x, y in pixels:
        cv2.circle(imagem, (int(x), int(y)), 2, 255, -1, cv2.LINE_AA)
    return imagem


def conv_relu_pool(entrada, peso, vies):
    canais, altura, largura = entrada.shape
    borda = np.pad(entrada, ((0, 0), (1, 1), (1, 1)))
    janelas = sliding_window_view(borda, (3, 3), axis=(1, 2))
    blocos = janelas.transpose(1, 2, 0, 3, 4).reshape(altura * largura, canais * 9)
    ativacao = np.maximum(blocos @ peso.reshape(len(peso), -1).T + vies, 0)
    return ativacao.reshape(altura // 2, 2, largura // 2, 2, len(vies)).max(axis=(1, 3)).transpose(2, 0, 1)


class CNN:
    classes = GESTOS

    def __init__(self, arquivo=PESOS_CNN):
        with np.load(arquivo, allow_pickle=False) as dados:
            self.pesos = tuple(dados[chave] for chave in ("w1", "b1", "w2", "b2", "w3", "b3"))
        self.imagem = None

    def probabilidades(self, entrada):
        self.imagem = rasterizar(entrada)
        w1, b1, w2, b2, w3, b3 = self.pesos
        ativacao = conv_relu_pool(self.imagem[None].astype(np.float32) / 255, w1, b1)
        ativacao = conv_relu_pool(ativacao, w2, b2)
        # Média em células 2×2, preservando a posição das partes da mão.
        resumo = ativacao.reshape(len(b2), 8, 2, 8, 2).mean(axis=(2, 4))
        logits = w3 @ resumo.ravel() + b3
        scores = np.exp(logits - logits.max())
        return scores / scores.sum()
