"""Mão no Jogo: colete estrelas em 20 s movendo o personagem por gestos ou pelas setas."""

import argparse
import ctypes
import math
import random
import time
import unicodedata
from contextlib import ExitStack
from pathlib import Path
from textwrap import wrap

import cv2
import numpy as np

from cnn import CNN, PESOS_CNN
from gestos import MODELO, ControleGestos
from treinador import solicitar_dica

LARGURA, ALTURA = 960, 540
CHAO = 430
RAIO_JOGADOR = 32
RAIO_ESTRELA = 22
VELOCIDADE = 420  # pixels por segundo
DURACAO = 20.0
DIRECAO = {"esquerda": -1, "direita": 1, "parar": 0}
JANELA = "Pedra Perdida"


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
        self.segundos_sem_mao = 0.0
        self.segundos_teclado = 0.0
        self.trocas_de_direcao = 0
        self.ultima_direcao = 0

    def registrar(self, dt, eixo, estado):
        intervalo = min(dt, self.restante)
        if estado == "teclado":
            self.segundos_teclado += intervalo
        if estado == "sem_mao":
            self.segundos_sem_mao += intervalo
        if estado == "reconhecido" and eixo:
            if self.ultima_direcao == -eixo:
                self.trocas_de_direcao += 1
            self.ultima_direcao = eixo

    def dados_da_rodada(self):
        return {
            "pontos": self.pontos,
            "segundos_sem_mao": round(self.segundos_sem_mao, 1),
            "segundos_teclado": round(self.segundos_teclado, 1),
            "trocas_de_direcao": self.trocas_de_direcao,
        }

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
        dt = min(dt, self.restante)
        self.restante -= dt
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
    else:
        escrever(tela, "Camera indisponivel - modo teclado", (470, 38), .48, (90, 190, 255), 1)

    return tela


def desenhar_entrada_cnn(tela, imagem):
    if imagem is None:
        return
    pixels = cv2.resize(imagem, (96, 96), interpolation=cv2.INTER_NEAREST)
    tela[29:125, 593:689] = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    escrever(tela, "Entrada CNN", (593, 20), 0.46, espessura=1)


def desenhar_resultado(tela, partida, resposta, treinador):
    cv2.rectangle(tela, (138, 168), (822, 477), (38, 26, 19), -1)
    cv2.rectangle(tela, (138, 168), (822, 477), (127, 93, 65), 1)
    escrever(tela, f"Fim de jogo  |  {partida.pontos} pontos", (172, 218), 1.0)
    dados = partida.dados_da_rodada()
    metricas = (f"Sem mao: {dados['segundos_sem_mao']:.1f} s   |   "
                f"Trocas: {dados['trocas_de_direcao']}   |   "
                f"Teclado: {dados['segundos_teclado']:.1f} s")
    escrever(tela, metricas, (172, 255), .52, (215, 207, 188), 1)
    cv2.line(tela, (172, 276), (788, 276), (127, 93, 65), 1)
    if resposta is not None and resposta.done():
        dica, fonte = resposta.result()
        escrever(tela, sem_acentos(fonte), (172, 310), .60, (118, 221, 255), 1)
        for i, linha in enumerate(wrap(sem_acentos(dica), width=64)):
            escrever(tela, linha, (172, 341 + 24 * i), .58, espessura=1)
    else:
        mensagem = "Gerando dica da rodada..." if treinador else "Treinador desativado."
        escrever(tela, mensagem, (172, 341), .59, espessura=1)
    escrever(tela, "Espaco ou R: jogar de novo    Esc: sair", (172, 447), .53,
             (215, 207, 188), 1)


def sem_acentos(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def tela_inicial():
    iniciado = False
    cursor_seta = None
    if hasattr(ctypes, "windll"):
        user32 = ctypes.windll.user32
        user32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.LoadCursorW.restype = ctypes.c_void_p
        user32.SetCursor.argtypes = [ctypes.c_void_p]
        cursor_seta = user32.LoadCursorW(None, ctypes.c_void_p(32512))

    def clicar(evento, x, y, flags, dados):
        nonlocal iniciado
        if cursor_seta:
            user32.SetCursor(cursor_seta)
        if evento == cv2.EVENT_LBUTTONUP and 330 <= x <= 630 and 290 <= y <= 360:
            iniciado = True

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(JANELA, clicar)
    while not iniciado:
        tela = np.full((ALTURA, LARGURA, 3), (70, 40, 25), np.uint8)
        escrever(tela, "Mao no Jogo", (290, 145), 1.6)
        escrever(tela, "Use gestos para coletar estrelas!", (245, 220), .7, espessura=1)
        cv2.rectangle(tela, (330, 290), (630, 360), (50, 150, 70), -1)
        escrever(tela, "INICIAR JOGO", (365, 336), .8)
        escrever(tela, "Clique em iniciar ou pressione Enter", (260, 420), .6, espessura=1)
        cv2.imshow(JANELA, tela)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (13, 32):
            iniciado = True
        elif tecla == 27 or cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
            return False
    return True


def jogar(controle, treinador=True):
    """Laço do jogo. controle=None usa o teclado."""
    partida = Partida()
    anterior = time.monotonic()
    resposta = None
    with ExitStack() as recursos:
        if controle is not None:
            recursos.enter_context(controle)
        while True:
            if controle is None:
                quadro, eixo = None, eixo_teclado()
                gesto = {-1: "esquerda", 1: "direita", 0: "parar"}[eixo] + " (teclado)"
                reconhecido = eixo != 0
                estado = "teclado"
            else:
                leitura, quadro = controle.ler()
                estado = leitura.estado
                if estado == "falha_camera":
                    recursos.close()
                    controle = None
                    print("A webcam parou de enviar quadros. Use as setas para continuar.")
                    quadro, eixo, gesto, reconhecido = None, 0, "parar (teclado)", False
                else:
                    eixo = DIRECAO[leitura.comando]
                    gesto = f"{leitura.comando} ({leitura.estado})"
                    reconhecido = leitura.estado == "reconhecido"
            agora = time.monotonic()
            dt = agora - anterior
            partida.registrar(dt, eixo, estado)
            partida.atualizar(dt, eixo)
            anterior = agora

            tela = desenhar_tela(partida, gesto, reconhecido, quadro)
            if controle is not None and isinstance(controle.classificador, CNN):
                desenhar_entrada_cnn(tela, controle.classificador.imagem)
            if partida.acabou:
                if treinador and resposta is None:
                    resposta = solicitar_dica(partida.dados_da_rodada())
                desenhar_resultado(tela, partida, resposta, treinador)
            cv2.imshow(JANELA, tela)
            tecla = cv2.waitKey(1 if controle is not None else 15) & 0xFF
            if tecla == 27 or cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                return
            if partida.acabou and tecla in (ord(" "), ord("r")):
                partida.reiniciar()
                resposta = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--teclado", action="store_true", help="joga com as setas, sem webcam")
    parser.add_argument("--modelo", type=Path, help="pesos da CNN (.npz) ou da MLP (.json)")
    parser.add_argument("--mlp", action="store_true", help="usa a MLP original")
    parser.add_argument("--sem-treinador", action="store_true", help="desliga o treinador")
    parser.add_argument("--sem-voz", action="store_true", help="silencia a voz do Mestre Granito")
    parser.add_argument("--semente", type=int, help="abre o mapa com a semente informada")
    args = parser.parse_args()
    from aventura import executar
    executar(args)


if __name__ == "__main__":
    main()
