"""Renderização da aventura em Full HD sobre coordenadas lógicas de 960 × 540."""

from pathlib import Path
from functools import lru_cache
import math

import cv2
import numpy as np

from jogo import ALTURA, LARGURA


# Paletas BGR, da rocha escura ao mineral iluminado.
AMBIENTES = (
    ('Portao das Profundezas', (32, 33, 45), (100, 151, 188), (149, 183, 213)),
    ('Lago das Lanternas', (36, 43, 29), (135, 144, 46), (218, 230, 109)),
    ('Vale das Ametistas', (66, 32, 56), (185, 87, 153), (255, 177, 249)),
    ('Rio de Magma', (34, 25, 46), (24, 74, 218), (87, 213, 255)),
    ('Coracao dos Geodos', (46, 28, 51), (157, 76, 142), (232, 171, 248)),
    ('Jardim Bioluminescente', (38, 45, 23), (132, 118, 25), (214, 245, 114)),
    ('Ossario de Pedra', (33, 40, 52), (105, 166, 212), (167, 217, 245)),
    ('Abismo de Obsidiana', (51, 28, 26), (148, 77, 46), (255, 200, 56)),
)
ATIVOS = Path(__file__).resolve().parent / 'assets'


def misturar(a, b, proporcao):
    return tuple(round(x + (y - x) * proporcao) for x, y in zip(a, b))


class Tela:
    FONTE = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, escala=2):
        self.escala = escala
        self.pixels = np.empty((ALTURA * escala, LARGURA * escala, 3), np.uint8)

    def ponto(self, posicao):
        x, y = posicao
        return round(x * self.escala), round(y * self.escala)

    def traco(self, espessura):
        return espessura if espessura < 0 else max(1, round(espessura * self.escala))

    def retangulo(self, inicio, fim, cor, espessura=-1):
        cv2.rectangle(self.pixels, self.ponto(inicio), self.ponto(fim), cor,
                      self.traco(espessura), cv2.LINE_AA)

    def linha(self, inicio, fim, cor, espessura=1):
        cv2.line(self.pixels, self.ponto(inicio), self.ponto(fim), cor,
                 self.traco(espessura), cv2.LINE_AA)

    def cartao(self, inicio, fim):
        x, y = inicio
        direita, baixo = fim
        self.retangulo((x + 3, y + 4), (direita + 3, baixo + 4), (20, 17, 22))
        self.retangulo(inicio, fim, (37, 30, 37))
        self.retangulo(inicio, fim, (101, 99, 129), 1)
        self.linha((x + 1, y + 1), (min(direita, x + 53), y + 1), (119, 195, 225), 2)

    def circulo(self, centro, raio, cor, espessura, tipo=cv2.LINE_AA):
        cv2.circle(self.pixels, self.ponto(centro), round(raio * self.escala), cor,
                   self.traco(espessura), tipo)

    def elipse(self, centro, eixos, angulo, inicio, fim, cor, espessura, tipo=cv2.LINE_AA):
        cv2.ellipse(self.pixels, self.ponto(centro), self.ponto(eixos), angulo, inicio, fim,
                    cor, self.traco(espessura), tipo)

    def poligono(self, pontos, cor):
        vertices = np.rint(np.asarray(pontos) * self.escala).astype(np.int32)
        cv2.fillPoly(self.pixels, [vertices], cor, cv2.LINE_AA)

    def texto(self, texto, posicao, escala=.8, cor=(244, 238, 224), espessura=2):
        cv2.putText(self.pixels, texto, self.ponto(posicao), self.FONTE,
                    escala * self.escala, cor, self.traco(espessura), cv2.LINE_AA)

    def largura_texto(self, texto, escala, espessura=1):
        tamanho, _ = cv2.getTextSize(texto, self.FONTE, escala * self.escala,
                                     self.traco(espessura))
        return tamanho[0] / self.escala

    def imagem(self, imagem, x, y, largura, altura, interpolacao=cv2.INTER_AREA):
        x0, y0 = self.ponto((x, y))
        x1, y1 = self.ponto((x + largura, y + altura))
        self.pixels[y0:y1, x0:x1] = cv2.resize(imagem, (x1 - x0, y1 - y0),
                                               interpolation=interpolacao)


def carregar_cavernas(tela):
    caminho = ATIVOS / 'caverna-panorama.png'
    quadro = cv2.imdecode(np.fromfile(str(caminho), np.uint8), cv2.IMREAD_COLOR)
    if quadro is None:
        raise ValueError(f'Fundo de caverna ilegível: {caminho}')
    altura, largura = tela.pixels.shape[:2]
    largura_fundo = round(largura * 1.30)
    fator = max(largura_fundo / quadro.shape[1], altura / quadro.shape[0])
    redimensionado = cv2.resize(quadro, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)
    y = (redimensionado.shape[0] - altura) // 2
    x = (redimensionado.shape[1] - largura_fundo) // 2
    base = redimensionado[y:y + altura, x:x + largura_fundo]
    fundos = {}
    for arquivo in (ATIVOS / 'cenarios').glob('[0-9][0-9]-*.png'):
        indice = int(arquivo.name[:2]) - 1
        if indice in fundos:
            raise ValueError(f'Mais de um cenário para a fase {indice + 1}.')
        fundos[indice] = arquivo
    if set(fundos) != set(range(len(AMBIENTES))):
        raise ValueError('São necessários os cenários numerados de 01 a 08.')
    berco = ATIVOS / '00-O-Berco-das-Pedras-Tutorial.png'
    fundos['tutorial'] = berco if berco.is_file() else ATIVOS / 'berco-das-pedras.png'
    menu = cv2.addWeighted(base, .78, np.full_like(base, (29, 28, 35)), .22, 0)
    return fundos, menu


@lru_cache(maxsize=2)
def montar_faixa(caminho, largura, altura):
    arte = cv2.imdecode(np.fromfile(str(caminho), np.uint8), cv2.IMREAD_COLOR)
    if arte is None:
        raise ValueError(f'Cenário ilegível: {caminho}')
    fator = max(largura / arte.shape[1], 1.16 * altura / arte.shape[0])
    tamanho = round(arte.shape[1] * fator), round(arte.shape[0] * fator)
    arte = cv2.resize(arte, tamanho, interpolation=cv2.INTER_AREA if fator < 1 else cv2.INTER_CUBIC)[:altura]
    # O chão pintado fica abaixo do percurso. A sombra separa a arte das plataformas jogáveis.
    y = np.linspace(0, 1, altura, dtype=np.float32)
    sombra = (.78 - .42 * np.clip((y - .68) / .32, 0, 1) ** 2)[:, None, None]
    arte = np.rint(arte * sombra).astype(np.uint8)
    espelho = arte[:, ::-1]
    sobreposicao = max(2, round(largura * .09))
    alfa = np.linspace(0, 1, sobreposicao, dtype=np.float32)[None, :, None]
    alfa = alfa * alfa * (3 - 2 * alfa)
    virar = np.rint(arte[:, -sobreposicao:] * (1 - alfa) + espelho[:, :sobreposicao] * alfa)
    voltar = np.rint(espelho[:, -sobreposicao:] * (1 - alfa) + arte[:, :sobreposicao] * alfa)
    return np.concatenate((arte[:, sobreposicao:-sobreposicao], virar.astype(np.uint8),
                           espelho[:, sobreposicao:-sobreposicao], voltar.astype(np.uint8)), axis=1)


def copiar_faixa(destino, faixa, deslocamento):
    largura = destino.shape[1]
    inicio = deslocamento % faixa.shape[1]
    trecho = min(largura, faixa.shape[1] - inicio)
    destino[:, :trecho] = faixa[:, inicio:inicio + trecho]
    if trecho < largura:
        destino[:, trecho:] = faixa[:, :largura - trecho]


def desenhar_profundidade(tela, cavernas, fase, camera_x):
    fundos, menu = cavernas
    altura, largura = tela.pixels.shape[:2]
    if fase is not None:
        faixa = montar_faixa(fundos[fase], largura, altura)
        copiar_faixa(tela.pixels, faixa, round(camera_x * tela.escala * .72))
        return
    tela.pixels[:] = menu[:, :largura]
    for i in range(8):
        x = i * 180 - 110
        y = 24 + (i % 3) * 16
        tela.poligono([(x, 0), (x + 116, 0), (x + 77, y), (x + 63, y + 21),
                      (x + 45, y - 9)], (32, 30, 40))
        tela.poligono([(x + 116, 0), (x + 77, y), (x + 63, y + 21),
                      (x + 72, 0)], (43, 39, 50))


def desenhar_rocha(tela, inicio, fim, altura, base, camera_x, tom=0, ambiente=0):
    esquerda = max(0, inicio - camera_x)
    direita = min(LARGURA, fim - camera_x)
    if esquerda >= direita:
        return
    _, sombra, meio, luz = AMBIENTES[ambiente]
    tela.retangulo((esquerda, altura), (direita, base), sombra)
    for bloco in range(inicio, fim, 66):
        x = max(esquerda, bloco - camera_x)
        y = min(direita, bloco + 66 - camera_x)
        if x >= y:
            continue
        variacao = ((bloco % 7) - 3 + tom) / 100
        centro = x + (y - x) * (.42 + (bloco % 5) * .04)
        tela.poligono([(x, altura), (y, altura), (centro, base), (x, base)],
                     misturar(sombra, meio, .27 + variacao))
        tela.poligono([(x, altura), (centro, altura), (centro + (y - x) / 6, min(base, altura + 12)),
                      (x, min(base, altura + 8))], misturar(sombra, meio, .50 + variacao))
    tela.linha((esquerda, altura), (direita, altura), misturar(sombra, luz, .63))


def desenhar_rampa(tela, inicio, fim, base, topo, ambiente):
    _, sombra, meio, luz = AMBIENTES[ambiente]
    tela.poligono([(inicio, base), (fim, topo), (fim, topo + 16), (inicio, base + 16)], sombra)
    tela.poligono([(inicio, base + 4), (fim, topo + 4), (fim - 15, topo + 18)], meio)
    tela.linha((inicio, base), (fim, topo), luz, 2)
    for x in range(math.ceil(inicio + 16), math.floor(fim), 24):
        y = base - (base - topo) * (x - inicio) / (fim - inicio)
        tela.linha((x, y + 4), (x + 5, y + 12), sombra, 2)


def desenhar_geiser(tela, x, y, fase, ciclo, duracao, aviso, ambiente):
    _, sombra, meio, luz = AMBIENTES[ambiente]
    ativo = fase < duracao
    aquecendo = fase > ciclo - aviso
    vapor = (101, 217, 255) if ambiente == 3 else (229, 214, 146)
    cor = vapor if ativo else (89, 205, 249) if aquecendo else meio
    tela.elipse((x, y + 5), (20, 8), 0, 0, 360, sombra, -1)
    tela.elipse((x, y + 1), (18, 5), 0, 0, 360, cor, 2)
    if ativo or aquecendo:
        altura = 104 if ativo else 26
        for i in range(6):
            subida = (i / 6 + fase * 1.6) % 1
            deriva = math.sin(i * 2.2 + fase * 7) * (6 + 10 * subida)
            tela.elipse((x + deriva, y - altura * subida), (8 + 6 * subida, 3 + 3 * subida),
                        0, 0, 360, misturar(cor, luz, subida * .35), -1)
        if ativo:
            tela.poligono([(x - 8, y), (x - 15, y - 67), (x, y - 103),
                           (x + 12, y - 64), (x + 9, y)], misturar(vapor, luz, .58))
    tela.cartao((x - 36, y + 17), (x + 36, y + 42))
    tela.texto('JA!' if ativo else 'GEISER', (x - (10 if ativo else 25), y + 30), .34, cor, 1)
    nivel = 1 - fase / duracao if ativo else (fase - duracao) / (ciclo - duracao)
    tela.retangulo((x - 29, y + 34), (x + 29, y + 37), sombra)
    tela.retangulo((x - 29, y + 34), (x - 29 + 58 * nivel, y + 37), cor)


def desenhar_inimigo(tela, x, piso, direcao, tempo):
    tela.elipse((x, piso - 1), (25, 5), 0, 0, 360, (22, 20, 27), -1)
    for i in (-1, 0, 1):
        perna = x + i * 10
        passo = math.sin(tempo * 13 + i * 2) * 5
        tela.linha((perna, piso - 13), (perna - 4 + passo, piso - 1), (82, 104, 159), 3)
    corpo = [(x - 22, piso - 7), (x - 24, piso - 20), (x - 13, piso - 33),
             (x + 6, piso - 35), (x + 22, piso - 24), (x + 23, piso - 9)]
    tela.poligono(corpo, (63, 53, 77))
    tela.poligono([corpo[1], corpo[2], corpo[3], (x + 4, piso - 15)], (98, 97, 141))
    tela.linha((x - 13, piso - 28), (x + 8, piso - 30), (120, 158, 206), 2)
    olho_x = x + 11 * direcao
    tela.linha((olho_x - 6, piso - 22), (olho_x + 6, piso - 20), (99, 192, 255), 3)


def desenhar_vida(tela, x, y, tempo):
    y += 4 * math.sin(tempo * 4)
    tela.circulo((x, y), 21, (67, 61, 163), 1)
    tela.circulo((x - 7, y - 4), 8, (120, 127, 254), -1)
    tela.circulo((x + 7, y - 4), 8, (120, 127, 254), -1)
    tela.poligono([(x - 14, y), (x, y + 14), (x + 14, y), (x, y - 6)], (120, 127, 254))
    tela.linha((x - 5, y), (x + 5, y), (240, 243, 250), 2)
    tela.linha((x, y - 5), (x, y + 5), (240, 243, 250), 2)
