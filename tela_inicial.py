"""Menu ilustrado e créditos de Mão no Jogo."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from narrativa import fonte, texto_acentuado

BOTOES = ('Play', 'Tutorial', 'Créditos')
ESQUERDA, TOPO, LARGURA, ALTURA, PASSO = 375, 210, 210, 55, 68
SAIR_CREDITOS = (390, 475, 570, 522)

REFERENCIAS = (
    ('Google AI Edge · MediaPipe Solutions', 'https://developers.google.com/edge/mediapipe'),
    ('NumPy Developers · NumPy 1.26 manual', 'https://numpy.org/doc/1.26/'),
    ('Ollama · Generate a chat message', 'https://docs.ollama.com/api/chat'),
    ('OpenCV · OpenCV 4.11.0 documentation', 'https://docs.opencv.org/4.11.0/'),
    ('PyTorch · PyTorch documentation', 'https://docs.pytorch.org/docs/stable/'),
    ('Qwen Team · Qwen2.5-0.5B-Instruct', 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct'),
    ('Scikit-learn Developers · Scikit-learn 1.6', 'https://scikit-learn.org/1.6/'),
)


def opcao_clicada(x, y):
    indice, altura = divmod(y - TOPO, PASSO)
    if ESQUERDA <= x <= ESQUERDA + LARGURA and 0 <= indice < len(BOTOES) and altura < ALTURA:
        return indice
    return None


def botao(tela, x, y, largura, altura, rotulo, selecionado, tamanho=23):
    borda = (142, 231, 255) if selecionado else (200, 158, 67)
    fundo = (43, 67, 76) if selecionado else (52, 42, 34)
    tela.poligono([(x - 1, y + 12), (x + 13, y + 1), (x + largura - 13, y + 1),
                   (x + largura + 1, y + 12), (x + largura + 1, y + altura - 7),
                   (x + largura - 13, y + altura + 4), (x + 13, y + altura + 4),
                   (x - 1, y + altura - 7)], (26, 23, 29))
    pontos = [(x, y + 10), (x + 12, y), (x + largura - 12, y),
              (x + largura, y + 10), (x + largura, y + altura - 10),
              (x + largura - 12, y + altura), (x + 12, y + altura), (x, y + altura - 10)]
    tela.poligono(pontos, fundo)
    for a, b in zip(pontos, pontos[1:] + pontos[:1]):
        tela.linha(a, b, borda, 2)
    if selecionado:
        tela.poligono([(x + 14, y + altura / 2), (x + 19, y + altura / 2 - 5),
                       (x + 24, y + altura / 2), (x + 19, y + altura / 2 + 5)], borda)
    escala = tela.escala
    texto_x = x + (largura - fonte(tamanho * escala, True).getlength(rotulo) / escala) / 2
    texto_acentuado(tela, rotulo, texto_x, y + (altura - tamanho) / 2 - 3,
                    tamanho, '#fff1c4' if selecionado else '#effbff', True)


def desenhar_inicio(tela, arte, opcao):
    tela.pixels[:] = arte
    for i, rotulo in enumerate(BOTOES):
        botao(tela, ESQUERDA, TOPO + i * PASSO, LARGURA, ALTURA, rotulo, opcao == i)
    tela.cartao((346, 414), (614, 440))
    texto_acentuado(tela, 'Indicador escolhe · Punho confirma', 366, 419, 14, '#e6f6ff')


@lru_cache(maxsize=2)
def faixa_creditos(escala):
    linhas = [(0, 'Diretor/CEO', 19, '#ffd78b', False),
              (29, 'Maycon Garcia Silva', 32, '#fff1db', True),
              (126, 'Referências', 25, '#ffd78b', True)]
    for i, (referencia, url) in enumerate(REFERENCIAS):
        linhas.extend([(178 + i * 63, referencia, 22, '#fff1db', True),
                       (207 + i * 63, url, 16, '#b8e2e8', False)])
    linhas += [(672, 'Designer', 19, '#ffd78b', False),
               (701, 'ChatGPT', 30, '#fff1db', True)]
    imagem = Image.new('RGBA', (750 * escala, 750 * escala))
    desenho = ImageDraw.Draw(imagem)
    for y, texto, tamanho, cor, destaque in linhas:
        desenho.text((375 * escala, y * escala), texto, anchor='mt', fill=cor,
                     font=fonte(tamanho * escala, destaque))
    return cv2.cvtColor(np.asarray(imagem), cv2.COLOR_RGBA2BGRA)


class Creditos:
    def __init__(self, arte):
        self.fundo = cv2.convertScaleAbs(arte, alpha=.25)
        self.tempo = 0.0

    def atualizar(self, dt):
        self.tempo += dt
        return 402 - self.tempo * 49 + 750 <= 91

    def desenhar(self, tela):
        tela.pixels[:] = self.fundo
        texto_acentuado(tela, 'Créditos', 426, 30, 28, '#ffe3ac', True)
        escala = tela.escala
        faixa = faixa_creditos(escala)
        posicao = round((402 - self.tempo * 49) * escala)
        inicio, fim = max(91 * escala, posicao), min(454 * escala, posicao + len(faixa))
        if inicio < fim:
            recorte = faixa[inicio - posicao:fim - posicao]
            destino = tela.pixels[inicio:fim, 105 * escala:855 * escala]
            linhas = np.arange(inicio, fim, dtype=np.float32) / escala
            visibilidade = np.clip(np.minimum(linhas - 91, 454 - linhas) / 24, 0, 1)
            alfa = recorte[:, :, 3:4] * (visibilidade[:, None, None] / 255)
            destino[:] = (recorte[:, :, :3] * alfa + destino * (1 - alfa)).astype(np.uint8)
        x, y, direita, base = SAIR_CREDITOS
        botao(tela, x, y, direita - x, base - y, 'Voltar', True, tamanho=18)
