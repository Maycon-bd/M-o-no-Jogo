"""Mão no Jogo: colete estrelas em 20 s movendo o personagem por gestos ou pelas setas."""

import argparse
import ctypes
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from gestos import MODELO, ControleGestos

LARGURA, ALTURA = 960, 540
CHAO = 430
RAIO_JOGADOR = 32
RAIO_ESTRELA = 22
VELOCIDADE = 420  # pixels por segundo
DURACAO = 20.0
DIRECAO = {"esquerda": -1, "direita": 1, "parar": 0}
JANELA = "Mao no Jogo | Esc sai"


class Partida:
    """Regras da rodada, iguais nos modos gesto e teclado."""

    def __init__(self, sorteio=None):
        self.sorteio = sorteio or random.Random()
        self.reiniciar()

    def reiniciar(self):
        self.x = LARGURA / 2
        self.pontos = 0
        self.restante = DURACAO
        self.estrela = self.nova_estrela()

    def nova_estrela(self):
        margem = RAIO_ESTRELA + 10
        while True:  # sempre há espaço livre: a tela tem 960 px e a exclusão, 400
            x = self.sorteio.uniform(margem, LARGURA - margem)
            if abs(x - self.x) > 200:
                return x

    @property
    def acabou(self):
        return self.restante <= 0

    def atualizar(self, dt, eixo):
        """Avança dt segundos reais com eixo -1, 0 ou 1. Parar zera a velocidade, o relógio segue."""
        if self.acabou:
            return
        self.restante = max(self.restante - dt, 0.0)
        antes = self.x
        self.x = min(max(self.x + eixo * VELOCIDADE * dt, RAIO_JOGADOR), LARGURA - RAIO_JOGADOR)
        # Testa o trecho percorrido, não só o ponto final, para um quadro lento não "pular" a estrela.
        alcance = RAIO_JOGADOR + RAIO_ESTRELA
        if min(antes, self.x) - alcance < self.estrela < max(antes, self.x) + alcance:
            self.pontos += 1
            self.estrela = self.nova_estrela()


def eixo_teclado():
    # shortcut: GetAsyncKeyState só existe no Windows (a bancada é Windows); em outro SO, usar pygame.
    tecla = ctypes.windll.user32.GetAsyncKeyState
    return bool(tecla(0x27) & 0x8000) - bool(tecla(0x25) & 0x8000)  # seta direita, seta esquerda


def escrever(tela, texto, posicao, escala=0.8, cor=(255, 255, 255), espessura=2):
    cv2.putText(tela, texto, posicao, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura, cv2.LINE_AA)


def desenhar_estrela(tela, cx, cy, raio):
    angulos = np.deg2rad(np.arange(10) * 36 - 90)
    raios = np.where(np.arange(10) % 2 == 0, raio, raio * 0.45)
    pontos = np.stack([cx + raios * np.cos(angulos), cy + raios * np.sin(angulos)], 1)
    cv2.fillPoly(tela, [pontos.astype(np.int32)], (0, 215, 255), cv2.LINE_AA)


def desenhar_tela(partida, gesto, reconhecido, quadro):
    tela = np.full((ALTURA, LARGURA, 3), (70, 40, 25), np.uint8)
    cv2.rectangle(tela, (0, CHAO + RAIO_JOGADOR), (LARGURA, ALTURA), (50, 120, 60), -1)
    desenhar_estrela(tela, partida.estrela, CHAO, RAIO_ESTRELA)
    x = int(partida.x)
    cv2.circle(tela, (x, CHAO), RAIO_JOGADOR, (60, 200, 255), -1, cv2.LINE_AA)
    for lado in (-11, 11):
        cv2.circle(tela, (x + lado, CHAO - 8), 5, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.ellipse(tela, (x, CHAO + 6), (12, 8), 0, 0, 180, (30, 30, 30), 2, cv2.LINE_AA)

    escrever(tela, f"Tempo: {math.ceil(partida.restante)} s", (20, 42))
    escrever(tela, f"Pontos: {partida.pontos}", (20, 82))
    escrever(tela, f"Gesto: {gesto}", (20, 122), cor=(90, 230, 90) if reconhecido else (190, 190, 190))
    dica = ("Indicador para o lado: anda | Palma aberta: para" if quadro is not None
            else "Setas esquerda e direita: anda | Solte: para")
    escrever(tela, dica, (20, ALTURA - 22), 0.65, espessura=1)

    if quadro is not None:
        tela[10:190, LARGURA - 250:LARGURA - 10] = cv2.resize(quadro, (240, 180))

    if partida.acabou:
        cv2.rectangle(tela, (230, 170), (730, 330), (30, 30, 30), -1)
        escrever(tela, "Fim de jogo!", (380, 220), 1.1)
        escrever(tela, f"Pontos: {partida.pontos}", (405, 265))
        escrever(tela, "Espaco ou R: jogar de novo | Esc: sair", (262, 308), 0.65, espessura=1)
    return tela


def jogar(controle):
    """Laço do jogo. controle=None usa o teclado."""
    partida = Partida()
    anterior = time.monotonic()
    while True:
        if controle:
            leitura, quadro = controle.ler()
            eixo = DIRECAO[leitura.comando]
            gesto, reconhecido = f"{leitura.comando} ({leitura.estado})", leitura.estado == "reconhecido"
        else:
            quadro, eixo = None, eixo_teclado()
            gesto, reconhecido = {-1: "esquerda", 1: "direita", 0: "parar"}[eixo] + " (teclado)", eixo != 0
        agora = time.monotonic()
        partida.atualizar(agora - anterior, eixo)
        anterior = agora

        cv2.imshow(JANELA, desenhar_tela(partida, gesto, reconhecido, quadro))
        tecla = cv2.waitKey(1 if controle else 15) & 0xFF
        if tecla == 27 or cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
            return
        if partida.acabou and tecla in (ord(" "), ord("r")):
            partida.reiniciar()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--teclado", action="store_true", help="joga com as setas, sem webcam")
    parser.add_argument("--modelo", type=Path, default=MODELO, help="pesos da MLP")
    args = parser.parse_args()

    controle = None
    if not args.teclado:
        try:
            controle = ControleGestos(args.camera, args.modelo)
        except (OSError, ImportError) as erro:
            print(f"{erro}\nSem webcam: abrindo no modo teclado.")
    try:
        jogar(controle)
    finally:
        if controle:
            controle.__exit__(None, None, None)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
