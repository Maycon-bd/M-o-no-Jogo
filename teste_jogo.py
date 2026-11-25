"""Confere as regras do jogo sem webcam: python teste_jogo.py"""

import random
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np

import jogo
import treinador
from gestos import Leitura
from jogo import DURACAO, LARGURA, RAIO_JOGADOR, VELOCIDADE, Partida, desenhar_tela

p = Partida(random.Random(1))
assert p.pontos == 0 and p.restante == DURACAO
assert abs(p.estrela - p.x) > 200, "a primeira estrela não nasce em cima do personagem"

# Parar: não anda, mas o relógio continua.
x = p.x
p.atualizar(1.0, 0)
assert p.x == x and p.restante == DURACAO - 1

# Limites da tela.
p.estrela = LARGURA - 40  # tira a estrela do caminho da esquerda
p.atualizar(5.0, -1)
assert p.x == RAIO_JOGADOR

# Coleta mesmo quando um quadro lento faz o personagem passar por cima da estrela.
p.x, p.estrela = 100.0, 500.0
p.atualizar(2.0, 1)
assert p.pontos == 1 and p.estrela != 500.0

# Um quadro atrasado usa apenas o tempo que restava, inclusive para pontuar.
p.x, p.estrela, p.restante = 300.0, 500.0, 0.125
p.atualizar(1.0, 1)
assert p.acabou and p.x == 300.0 + 0.125 * VELOCIDADE and p.pontos == 1
x, pontos = p.x, p.pontos
p.atualizar(1.0, -1)
assert (p.x, p.pontos) == (x, pontos)

p.x, p.estrela, p.restante = 300.0, 400.0, 0.125
p.atualizar(1.0, 1)
assert p.acabou and p.pontos == 2  # ainda alcançou a estrela dentro do tempo

# Reinício zera o placar e o tempo.
p.reiniciar()
assert p.pontos == 0 and p.restante == DURACAO and not p.acabou

# A tela desenha nos dois modos e na tela final.
assert desenhar_tela(p, "parar (teclado)", False, None).shape == (540, 960, 3)
assert desenhar_tela(p, "direita (reconhecido)", True, np.zeros((480, 640, 3), np.uint8)).shape == (540, 960, 3)
p.atualizar(100, 0)
desenhar_tela(p, "parar", False, None)

# Perder a câmera preserva a partida e libera o recurso antes de ler o teclado.
controle = MagicMock()
controle.ler.side_effect = [
    (Leitura("parar", "sem_mao"), np.zeros((480, 640, 3), np.uint8)),
    (Leitura("parar", "falha_camera"), None),
]
p = Partida(random.Random(1))
p.pontos = 4
quadros = []


def registrar(partida, gesto, reconhecido, quadro):
    quadros.append((partida.x, partida.restante, partida.pontos, gesto, quadro is None))


def pressionar_direita():
    controle.__exit__.assert_called_once()
    return 1


with (
    patch.object(jogo, "Partida", return_value=p),
    patch.object(jogo.time, "monotonic", side_effect=[0.0, 0.125, 0.25, 0.375]),
    patch.object(jogo, "eixo_teclado", side_effect=pressionar_direita),
    patch.object(jogo, "desenhar_tela", side_effect=registrar),
    patch.object(jogo.cv2, "imshow"),
    patch.object(jogo.cv2, "waitKey", side_effect=[-1, -1, 27]),
    patch.object(jogo.cv2, "getWindowProperty", return_value=1),
):
    jogo.jogar(controle)

assert quadros == [
    (LARGURA / 2, DURACAO - 0.125, 4, "parar (sem_mao)", False),
    (LARGURA / 2, DURACAO - 0.25, 4, "parar (teclado)", True),
    (LARGURA / 2 + 0.125 * VELOCIDADE, DURACAO - 0.375, 4, "direita (teclado)", True),
]
assert controle.ler.call_count == 2
controle.__exit__.assert_called_once()

# Os dados enviados ao SLM contam apenas o tempo que ainda pertence à rodada.
p.reiniciar()
p.estrela, p.restante = 900, 0.25
p.registrar(0.125, 1, "reconhecido")
p.registrar(0.125, -1, "reconhecido")
p.registrar(0.125, 0, "sem_mao")
resposta = Future()
resposta.set_result(("Mantenha a mão visível na câmera.", "SLM local"))

with (
    patch.object(jogo, "Partida", return_value=p),
    patch.object(jogo.time, "monotonic", side_effect=[0.0, 0.5, 0.625]),
    patch.object(jogo, "eixo_teclado", return_value=0),
    patch.object(jogo, "solicitar_dica", return_value=resposta) as solicitar,
    patch.object(jogo.cv2, "imshow"),
    patch.object(jogo.cv2, "waitKey", side_effect=[-1, 27]),
    patch.object(jogo.cv2, "getWindowProperty", return_value=1),
):
    jogo.jogar(None)

dados = {"pontos": 0, "segundos_sem_mao": 0.1, "segundos_teclado": 0.2, "trocas_de_direcao": 1}
solicitar.assert_called_once_with(dados)
with patch.object(treinador, "build_opener", side_effect=OSError("serviço indisponível")):
    dica, fonte = treinador.pedir_dica(dados)
    assert fonte == "Dica de contingência" and "seta" in dica

print("ok: regras, recuperação sem câmera e treinador conferidos")
