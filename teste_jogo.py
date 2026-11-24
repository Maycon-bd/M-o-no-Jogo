"""Confere as regras do jogo sem webcam: python teste_jogo.py"""

import random

from jogo import DURACAO, LARGURA, RAIO_JOGADOR, Partida, desenhar_tela

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

# Fim aos 20 s: nada mais muda.
p.atualizar(100, 1)
assert p.acabou and p.restante == 0
x, pontos = p.x, p.pontos
p.atualizar(1.0, -1)
assert (p.x, p.pontos) == (x, pontos)

# Reinício zera o placar e o tempo.
p.reiniciar()
assert p.pontos == 0 and p.restante == DURACAO and not p.acabou

# A tela desenha nos dois modos e na tela final.
import numpy as np
assert desenhar_tela(p, "parar (teclado)", False, None).shape == (540, 960, 3)
assert desenhar_tela(p, "direita (reconhecido)", True, np.zeros((480, 640, 3), np.uint8)).shape == (540, 960, 3)
p.atualizar(100, 0)
desenhar_tela(p, "parar", False, None)

print("ok: regras do jogo conferidas")
