"""Jogo de gestos com tutorial e desafios progressivos."""

import math
import ctypes
import random
import time
from bisect import bisect_right
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import cv2
import numpy as np

from cnn import CNN, PESOS_CNN_JOGO
from gestos import ControleGestos, MODELO
from jogo import ALTURA, DIRECAO, JANELA, LARGURA, sem_acentos, solicitar_dica
from voz_granito import VozGranito
from narrativa import (ABERTURA, ETAPAS_TREINO, FALAS_TREINO, Dialogo, Introducao, texto_acentuado)
from salvamento import Progresso
from tela_inicial import BOTOES, SAIR_CREDITOS, Creditos, desenhar_inicio, opcao_clicada
from visual_caverna import (AMBIENTES, Tela, carregar_cavernas, desenhar_rocha,
                            desenhar_profundidade,
                            desenhar_geiser, desenhar_inimigo, desenhar_rampa, desenhar_vida)

MAPA_RETRATO = (10, 55, 744, 419)
DISCOS = ((204, 237), (537, 380), (934, 221), (1508, 236),
          (1386, 604), (975, 730), (537, 731), (186, 640))
# Pontos centrais das trilhas na imagem, em pixels da arte original.
CURVAS = (
    ((253, 251), (297, 280), (274, 325), (289, 345), (339, 367),
     (372, 405), (416, 413), (472, 396)),
    ((622, 381), (698, 394), (769, 360), (795, 328), (801, 274), (835, 240), (875, 229)),
    ((1031, 227), (1094, 250), (1110, 295), (1153, 317),
     (1211, 308), (1279, 267), (1354, 220), (1425, 217)),
    ((1555, 267), (1574, 306), (1557, 365), (1582, 398), (1613, 446),
     (1630, 505), (1610, 562), (1563, 605), (1498, 623), (1438, 616)),
    ((1318, 625), (1268, 666), (1263, 725), (1228, 761),
     (1151, 764), (1083, 749), (1021, 737)),
    ((912, 740), (852, 754), (806, 790), (749, 790), (714, 752),
     (667, 728), (607, 727)),
    ((473, 735), (414, 746), (351, 755), (310, 733), (282, 684), (247, 659)),
)


def no_mapa(ponto):
    x, y, largura, altura = MAPA_RETRATO
    return x + ponto[0] * largura / 1672, y + ponto[1] * altura / 941


def amostrar_trilha(pontos):
    """Interpola a curva e mede seu comprimento para rolar a velocidade constante."""
    pontos = [np.array(no_mapa(ponto)) for ponto in pontos]
    estendidos = [pontos[0], *pontos, pontos[-1]]
    amostras = [pontos[0]]
    for a, b, c, d in zip(estendidos, estendidos[1:], estendidos[2:], estendidos[3:]):
        for passo in range(1, 9):
            t = passo / 8
            amostras.append(b + .5 * t * ((c - a) + t * ((2*a - 5*b + 4*c - d)
                            + t * (-a + 3*b - 3*c + d))))
    distancias = [0.0]
    for anterior, ponto in zip(amostras, amostras[1:]):
        distancias.append(distancias[-1] + math.dist(anterior, ponto))
    return amostras, distancias


ETAPAS = tuple(map(no_mapa, DISCOS))
TRILHAS = tuple(amostrar_trilha((DISCOS[i], *curva, DISCOS[i + 1]))
                for i, curva in enumerate(CURVAS))
NAVEGACAO = {'esquerda': -1, 'direita': 1}

ASSETS = Path(__file__).resolve().parent / 'assets'


def carregar_imagem(caminho, tamanho):
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if imagem is None or imagem.ndim != 3 or imagem.shape[2] != 4:
        raise ValueError(f'Imagem transparente invalida: {caminho}')
    return cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA)


def colar(tela, imagem, x, y):
    altura, largura = imagem.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(tela.shape[1], x + largura), min(tela.shape[0], y + altura)
    if x0 >= x1 or y0 >= y1:
        return
    recorte = imagem[y0-y:y1-y, x0-x:x1-x]
    alfa = recorte[:, :, 3:4].astype(np.float32) / 255
    tela[y0:y1, x0:x1] = (recorte[:, :, :3] * alfa + tela[y0:y1, x0:x1] * (1 - alfa)).astype(np.uint8)


def carregar_personagens(escala=2):
    arquivos = ('01-parado', '02-rolando', '03-pulando', '04-velocidade',
                '05-impacto', '06-susto-queda', '07-vitoria')
    sprites = {nome[3:]: carregar_imagem(ASSETS / 'rocha_sprites' / (nome + '.png'), (64 * escala, 64 * escala))
               for nome in arquivos}
    expressoes = ('01-ensinando', '02-alertando', '03-orgulhoso')
    retratos = {nome[3:]: carregar_imagem(ASSETS / 'mestre_granito_expressoes' / (nome + '.png'),
                                         (68 * escala, 68 * escala))
                for nome in expressoes}
    return sprites, retratos


def linhas_fala(tela, dica, largura=475):
    linhas = []
    linha = ''
    for palavra in sem_acentos(dica).split():
        candidata = f'{linha} {palavra}' if linha else palavra
        if tela.largura_texto(candidata, .39) <= largura:
            linha = candidata
            continue
        if linha:
            linhas.append(linha)
        linha = ''
        for letra in palavra:
            if tela.largura_texto(linha + letra, .39) > largura:
                linhas.append(linha)
                linha = ''
            linha += letra
    if linha:
        linhas.append(linha)
    return linhas


def duracao_fala(dica):
    return max(8.0, 2.0 + len(dica.split()) / 2.25)


def caixa_voz(lateral):
    return (775, 281, 824, 302) if lateral else (224, 53, 276, 74)


def desenhar_caixa_voz(tela, lateral, marcada):
    x, y, _, base = caixa_voz(lateral)
    cor = (135, 222, 182) if marcada else (173, 163, 178)
    tela.retangulo((x + 4, y + 4), (x + 16, base - 5), cor, 1)
    if marcada:
        tela.linha((x + 6, y + 10), (x + 9, y + 13), cor, 1)
        tela.linha((x + 9, y + 13), (x + 14, y + 7), cor, 1)
    tela.texto('Voz', (x + 22, y + 15), .36, cor, 1)


def desenhar_mestre(tela, imagem, dica, fonte, lateral=False, voz_ativa=True):
    if lateral:
        tela.cartao((765, 249), (949, 474))
        tela.texto('MESTRE GRANITO', (779, 267), .40, (100, 220, 255), 1)
        colar(tela.pixels, imagem, 822 * tela.escala, 273 * tela.escala)
        for i, linha in enumerate(linhas_fala(tela, dica, 157)):
            tela.texto(linha, (779, 357 + i * 14), .39, espessura=1)
        tela.texto(sem_acentos(fonte), (779, 460), .32, (180, 180, 180), 1)
        desenhar_caixa_voz(tela, lateral, voz_ativa)
        return
    linhas = linhas_fala(tela, dica)
    y_fonte = max(129, 88 + (len(linhas) - 1) * 13 + 14)
    tela.cartao((10, 48), (577, y_fonte + 9))
    colar(tela.pixels, imagem, 16 * tela.escala, 59 * tela.escala)
    tela.texto('MESTRE GRANITO', (90, 68), .42, (100, 220, 255), 1)
    for i, linha in enumerate(linhas):
        tela.texto(linha, (90, 88 + i * 13), .39, espessura=1)
    tela.texto(sem_acentos(fonte), (90, y_fonte), .32, (180, 180, 180), 1)
    desenhar_caixa_voz(tela, lateral, voz_ativa)


def punho_fechado(pontos):
    if pontos is None:
        return False
    # As quatro pontas ficam próximas ao punho quando os dedos estão dobrados.
    distancias = np.linalg.norm(pontos[:, :2] - pontos[0, :2], axis=1)
    return all(distancias[ponta] < distancias[base] * 1.2
               for ponta, base in ((8, 6), (12, 10), (16, 14), (20, 18)))


def gesto_da_mao(pontos, aceitar_para_baixo=False):
    if pontos is None:
        return None
    distancias = np.linalg.norm(pontos[:, :2] - pontos[0, :2], axis=1)
    dedos = ((6, 8), (10, 12), (14, 16), (18, 20))
    estendidos = tuple(distancias[ponta] > 1.25 * distancias[base] for base, ponta in dedos)
    if all(estendidos):
        return 'parar'
    if estendidos == (True, True, True, False):
        return 'tres'
    if estendidos != (True, False, False, False):
        return None
    dx, dy = pontos[8, :2] - pontos[5, :2]
    escala = np.linalg.norm(pontos[5, :2] - pontos[0, :2])
    if escala < 1 or math.hypot(dx, dy) < .65 * escala:
        return None
    if aceitar_para_baixo or dy <= 0 or dy <= abs(dx) * .75:
        eixo = eixo_indicador(pontos)
        if eixo:
            return 'esquerda' if eixo < 0 else 'direita'
        return 'vertical'
    return None


def eixo_indicador(pontos):
    """Converte a inclinação do indicador em um comando entre -1 e 1."""
    dx, dy = pontos[8, :2] - pontos[5, :2]
    angulo = math.degrees(math.atan2(abs(dx), abs(dy)))
    intensidade = max(0.0, min(1.0, (angulo - 10) / 65))
    return math.copysign(intensidade, dx)


def direcao_menu_inicial(pontos):
    """O indicador acompanha a disposição vertical dos botões ilustrados."""
    dx, dy = pontos[8, :2] - pontos[5, :2]
    componente = dy if abs(dy) > abs(dx) else dx
    return 'direita' if componente > 0 else 'esquerda'


def iniciar_controle(args):
    caminho = args.modelo or (MODELO if args.mlp else PESOS_CNN_JOGO)
    classificador = CNN(caminho) if caminho.suffix.lower() == '.npz' else None
    return ControleGestos(args.camera, caminho, classificador=classificador)


def painel(tela, titulo, linhas):
    tela.cartao((115, 150), (845, 460))
    tela.texto(titulo, (145, 205), 1.05, (189, 226, 248))
    tela.linha((145, 219), (815, 219), (78, 70, 92))
    for i, linha in enumerate(linhas):
        tela.texto(linha, (145, 250 + i * 34), .53, (225, 225, 225), 1)


class RochaNoMapa:
    def __init__(self):
        self.no = self.destino = 0
        self.fracao = self.angulo = 0.0

    def atualizar(self, dt, vetor, liberada, passo_teclado=0):
        if self.no == self.destino and passo_teclado:
            vizinho = self.no + passo_teclado
            if 0 <= vizinho <= liberada:
                self.destino = vizinho
        elif self.no == self.destino and vetor is not None:
            x, y = ETAPAS[self.no]
            melhor = .46
            for vizinho in (self.no - 1, self.no + 1):
                if 0 <= vizinho <= liberada:
                    a, b = ETAPAS[vizinho]
                    alinhamento = ((a - x) * vetor[0] + (b - y) * vetor[1]) / math.hypot(a - x, b - y)
                    if alinhamento > melhor:
                        self.destino, melhor = vizinho, alinhamento
        if self.no != self.destino:
            x_anterior, _ = self.posicao()
            _, distancias = TRILHAS[min(self.no, self.destino)]
            total = distancias[-1]
            restante = (1 - self.fracao) * total
            distancia = min(restante, dt * 290)
            self.fracao = min(1.0, self.fracao + distancia / total)
            x, _ = self.posicao()
            self.angulo += math.copysign(math.degrees(distancia / 12), x - x_anterior)
            if distancia >= restante:
                self.no, self.fracao = self.destino, 0.0
                self.angulo %= 360

    def posicao(self):
        if self.no == self.destino:
            return ETAPAS[self.no]
        amostras, distancias = TRILHAS[min(self.no, self.destino)]
        distancia = (self.fracao if self.no < self.destino else 1 - self.fracao) * distancias[-1]
        i = max(0, min(len(distancias) - 2, bisect_right(distancias, distancia) - 1))
        t = (distancia - distancias[i]) / (distancias[i + 1] - distancias[i])
        return amostras[i] + (amostras[i + 1] - amostras[i]) * t


def desenhar_mapa(tela, partida, mapa, sprites_mapa, progresso, carregando, arte):
    tela.pixels[:] = (29, 23, 29)
    tela.imagem(arte, *MAPA_RETRATO, cv2.INTER_AREA)
    tela.retangulo((9, 54), (754, 474), (119, 98, 139), 1)
    tela.texto('MAPA DAS OITO CAVERNAS', (22, 34), .76, (189, 226, 248))
    tela.texto(f'Luzes {partida.pontos}  |  Vidas {partida.vidas}  |  Semente {partida.semente}',
               (628, 32), .40, espessura=1)
    for indice, inicio, fim in ((2, 4, 7), (3, 3, 10)):
        if indice < partida.fase:
            for ponto in CURVAS[indice][inicio:fim]:
                x, y = no_mapa(ponto)
                tela.elipse((x, y), (3, 2), 0, 0, 360, (150, 197, 232), -1)
    for i, (x, y) in enumerate(ETAPAS):
        cor = (114, 221, 255) if i == partida.fase else (122, 234, 173) if i < partida.fase else (183, 172, 180)
        if i <= partida.fase:
            tela.elipse((x, y), (18, 9), 0, 0, 360, cor, 2)
        tela.circulo((x + 13, y + 9), 8, (40, 29, 38), -1)
        tela.texto(str(i + 1), (x + 9.5, y + 12), .31, cor, 1)
    x, y = mapa.posicao()
    rolando = mapa.no != mapa.destino
    imagem = sprites_mapa['rolando' if rolando else 'parado']
    lado = imagem.shape[0]
    if rolando:
        matriz = cv2.getRotationMatrix2D((lado / 2, lado / 2), -mapa.angulo, 1)
        imagem = cv2.warpAffine(imagem, matriz, (lado, lado))
    tela.elipse((x, y - 2), (15, 5), 0, 0, 360, (24, 20, 24), -1)
    colar(tela.pixels, imagem, round(x * tela.escala - lado / 2), round((y - 15) * tela.escala - lado / 2))
    selecionada = mapa.destino
    nome = AMBIENTES[selecionada][0]
    acao = 'Revisitar' if selecionada < partida.fase or partida.venceu else 'Entrar'
    mensagem = 'Preparando a camera...' if carregando else f'{acao} | {selecionada + 1}. {nome}'
    tela.cartao((10, 485), (949, 529))
    tela.texto(mensagem, (23, 513), .57, espessura=1)
    tela.texto('Aponte ou use as setas. Punho fechado, Enter ou clique para entrar.',
               (451, 510), .38, espessura=1)
    if progresso > 0:
        tela.retangulo((23, 521), (23 + 900 * min(progresso, 1), 524), (105, 208, 165))


def painel_camera(tela, quadro, controle, leitura, gesto, carregando, lateral=False):
    imagem = getattr(getattr(controle, 'classificador', None), 'imagem', None)
    acao = {'punho': 'PUNHO', 'parar': 'PALMA', 'tres': 'MENU'}.get(gesto)
    if lateral:
        tela.cartao((765, 55), (949, 242))
        tela.texto('CAMERA', (779, 68), .34, espessura=1)
        if quadro is not None:
            tela.imagem(quadro, 785, 72, 144, 108)
        else:
            tela.retangulo((785, 72), (929, 180), (24, 22, 29))
            tela.texto('Preparando' if carregando else 'Indisponivel', (800, 130), .40, espessura=1)
        tela.retangulo((779, 189), (826, 235), (24, 22, 29))
        if imagem is not None:
            tela.imagem(cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR), 779, 189, 46, 46, cv2.INTER_NEAREST)
        tipo = 'MLP' if controle is not None and not isinstance(controle.classificador, CNN) else 'CNN'
        tela.texto(f'LEITURA {tipo}', (833, 201), .29, espessura=1)
        texto = 'Aguardando'
        if leitura is not None:
            texto = ('Sem mao' if leitura.estado == 'sem_mao' else
                     f'{leitura.comando} {leitura.confianca:.0%}')
        tela.texto(texto, (833, 216), .31, espessura=1)
        if acao:
            tela.texto(acao, (833, 231), .30, (80, 220, 255), 1)
        return
    tela.cartao((600, 2), (953, 143))
    tela.texto('CAMERA', (786, 18), .38, espessura=1)
    tela.texto('ENTRADA CNN', (648, 18), .38, espessura=1)
    if quadro is not None:
        tela.imagem(quadro, 786, 22, 136, 102)
    else:
        status = 'Preparando' if carregando else 'Indisponivel'
        tela.texto(status, (790, 76), .43, (190, 190, 190), 1)

    if imagem is not None:
        pixels = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        tela.imagem(pixels, 655, 23, 96, 96, cv2.INTER_NEAREST)
    else:
        mensagem = 'Sem mao' if quadro is not None else 'Aguardando'
        if controle is not None and not isinstance(controle.classificador, CNN):
            mensagem = 'Modo MLP'
        tela.texto(mensagem, (640, 73), .4, (190, 190, 190), 1)

    if leitura is None:
        texto = 'Leitura: aguardando'
    elif leitura.estado == 'sem_mao':
        texto = 'Leitura: sem mao'
    else:
        texto = f'CNN: {leitura.comando} {leitura.confianca:.0%} ({leitura.estado})'
        if controle is not None and not isinstance(controle.classificador, CNN):
            texto = f'MLP: {leitura.comando} {leitura.confianca:.0%} ({leitura.estado})'
    tela.texto(texto, (610, 138), .36, (220, 220, 220), 1)
    if acao:
        tela.texto(acao, (605, 37), .32, (80, 220, 255), 1)


def executar(args):
    from jogo import eixo_teclado
    tela = Tela()
    arte_titulo = cv2.imdecode(np.fromfile(ASSETS / 'tela-inicial.png', np.uint8), cv2.IMREAD_COLOR)
    if arte_titulo is None:
        raise ValueError('Imagem da tela inicial ilegível.')
    arte_titulo = cv2.resize(arte_titulo, (tela.pixels.shape[1], tela.pixels.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    cavernas = carregar_cavernas(tela)
    sprites, mestre = carregar_personagens(tela.escala)
    arte_mapa = cv2.imdecode(np.fromfile(ASSETS / 'mapa-oito-cavernas.png', np.uint8), cv2.IMREAD_COLOR)
    if arte_mapa is None:
        raise ValueError('Imagem do mapa das oito cavernas ilegivel.')
    lado_mapa = 33 * tela.escala
    sprites_mapa = {nome: cv2.resize(sprites[nome], (lado_mapa, lado_mapa), interpolation=cv2.INTER_AREA)
                    for nome in ('parado', 'rolando')}
    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    clique = [None]
    def mouse(evento, x, y, flags, dados):
        if hasattr(__import__('ctypes'), 'windll'):
            import ctypes
            user32 = ctypes.windll.user32
            user32.LoadCursorW.restype = ctypes.c_void_p
            user32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            user32.SetCursor.argtypes = [ctypes.c_void_p]
            user32.SetCursor(user32.LoadCursorW(None, ctypes.c_void_p(32512)))
        if evento == cv2.EVENT_LBUTTONUP:
            clique[0] = (x // tela.escala, y // tela.escala)
    cv2.setMouseCallback(JANELA, mouse)
    executor = ThreadPoolExecutor(max_workers=1)
    voz = VozGranito(ativa=not getattr(args, 'sem_voz', False))
    futuro = None if args.teclado else executor.submit(iniciar_controle, args)
    controle = None
    erro_camera = None
    estado = 'inicio'
    tempo_gesto = 0.0
    anterior = time.monotonic()
    partida = None
    campanha = None
    mapa = RochaNoMapa()
    resposta = None
    dica_mestre = 'Aponte para escolher e feche a mão para confirmar.'
    fonte_mestre = 'Mestre Granito'
    expressao_mestre = 'ensinando'
    perigo_anterior = None
    explicou_geiser = False
    proxima_dica = 0.0
    tempo_leitura = 0.0
    resposta_durante = None
    bloqueio = False
    gesto_candidato = None
    tempo_candidato = 0.0
    menu_indice = 0
    menu_ultimo_gesto = None
    proximo_passo_menu = 0.0
    menu_confirmar = False
    origem_pausa = 'jogo'
    opcao_inicio = 0
    ultimo_lado_inicio = 0
    modo_semente = getattr(args, 'semente', None) is not None
    save = None if modo_semente else Progresso.ler(len(ETAPAS), len(ETAPAS_TREINO))
    intro = None
    creditos = None
    tutorial_avulso = False
    rever_intro = False
    dialogo = None
    interacao = 0
    ocioso = 0.0
    proxima_fala_contextual = 0.0
    volta_gameover = 0.0
    queda_repetida_pendente = False

    def dizer(dica, expressao='ensinando', fonte='Mestre Granito'):
        nonlocal dica_mestre, expressao_mestre, fonte_mestre, tempo_leitura
        dica_mestre, expressao_mestre, fonte_mestre = dica, expressao, fonte
        tempo_leitura = duracao_fala(dica)
        voz.falar(dica)

    def gravar():
        if save is not None and not modo_semente:
            try:
                save.gravar()
            except OSError as erro:
                print(f'Não foi possível gravar o progresso: {erro}')
                dizer('Não foi possível gravar o progresso no disco.', 'alertando')

    def titulo():
        nonlocal estado, partida, campanha, tempo_gesto, bloqueio, opcao_inicio, tutorial_avulso
        estado, partida, campanha = 'inicio', None, None
        tempo_gesto, bloqueio, opcao_inicio = 0.0, True, 0
        tutorial_avulso = False
        voz.interromper()

    def iniciar_abertura(rever=False):
        nonlocal estado, intro, rever_intro, bloqueio, tempo_gesto
        estado, intro, rever_intro = 'intro', Introducao(rever=rever), rever
        bloqueio, tempo_gesto = True, 0.0
        voz.interromper()

    def iniciar_partida(semente=None, continuar=False):
        nonlocal partida, campanha, mapa, estado, tempo_gesto, bloqueio
        nonlocal resposta, resposta_durante, proxima_dica
        if continuar and not save.mapa_liberado(len(ETAPAS_TREINO)):
            abrir_tutorial(save.tutorial)
            return
        partida = PartidaAventura(save.semente if continuar else semente)
        if continuar:
            partida.fase = save.fase
            partida.carregar_fase()
            fase = partida.fases[partida.fase]
            if save.checkpoint in fase.get('checkpoints', ()):
                partida.x = partida.checkpoint = save.checkpoint
                partida.camera_x = min(max(0, partida.x - 320), fase['chao'][-1][1] - LARGURA)
            partida.pontos = save.luzes
            partida.venceu = save.venceu
            partida.luzes = [p for i, p in enumerate(partida.luzes) if i not in save.luzes_recolhidas]
            partida.vidas_extras = [p for i, p in enumerate(partida.vidas_extras) if i not in save.vidas_recolhidas]
        campanha = None
        mapa = RochaNoMapa()
        mapa.no = mapa.destino = partida.fase
        estado, tempo_gesto, bloqueio = 'mapa', 0.0, True
        clique[0] = None
        resposta = resposta_durante = None
        proxima_dica = 0.0
        dizer('Aponte para escolher a caverna. Feche a mão para entrar.')

    def gravar_campanha():
        if save is None or campanha is not None:
            return
        save.fase, save.checkpoint, save.luzes = partida.fase, int(partida.checkpoint), partida.pontos
        save.venceu |= partida.venceu
        fase = partida.fases[partida.fase]
        save.luzes_recolhidas = [i for i, p in enumerate(fase['luzes']) if p not in partida.luzes]
        save.vidas_recolhidas = [i for i, p in enumerate(fase.get('vidas_extras', ())) if p not in partida.vidas_extras]
        gravar()

    def entrar_fase():
        nonlocal partida, campanha, estado, tempo_gesto, bloqueio, proxima_dica, resposta_durante
        nonlocal perigo_anterior, explicou_geiser, ocioso, proxima_fala_contextual, queda_repetida_pendente
        if mapa.no < partida.fase or partida.venceu:
            campanha = partida
            partida = PartidaAventura(campanha.semente, campanha.fases)
            partida.fase = mapa.no
            partida.carregar_fase()
        partida.salto_anterior = gesto == 'parar'
        estado, tempo_gesto, bloqueio = 'jogo', 0.0, gesto == 'tres'
        proxima_dica = 0.0
        resposta_durante = None
        perigo_anterior = None
        explicou_geiser = False
        ocioso, proxima_fala_contextual = 0.0, 0.0
        queda_repetida_pendente = False
        dizer('Pegue impulso e abra a mão para saltar.')

    def iniciar_dialogo_treino():
        nonlocal dialogo, interacao, bloqueio, tempo_gesto
        bloqueio, tempo_gesto, interacao = True, 0.0, 0
        etapa = partida.etapa
        if etapa == len(ETAPAS_TREINO):
            dialogo = None
            partida.expressao = 'vitoria'
            dizer('Tutorial concluído! Volte ao título para jogar o modo História.' if tutorial_avulso else
                  'Tutorial concluído! Sua jornada começa agora. O Portão das Profundezas está liberado.', 'orgulhoso')
            return
        dialogo = Dialogo(FALAS_TREINO[etapa])
        partida.expressao = {0: 'susto-queda', 2: 'vitoria', 3: 'vitoria', 4: 'velocidade', 5: 'susto-queda'}.get(etapa)
        dizer(dialogo.atual, 'orgulhoso' if etapa in (2, 3, 4, 5) else 'ensinando')

    def abrir_tutorial(etapa=0):
        nonlocal partida, estado
        partida = Treino(etapa)
        partida.avulso = tutorial_avulso
        estado = 'tutorial'
        iniciar_dialogo_treino()

    def concluir_licao():
        partida.etapa += 1
        if not tutorial_avulso and save is not None and partida.etapa > save.tutorial:
            save.tutorial = partida.etapa
            gravar()
        iniciar_dialogo_treino()

    def avancar_dialogo():
        nonlocal dialogo, bloqueio, tempo_gesto
        bloqueio, tempo_gesto = True, 0.0
        fala = dialogo.avancar()
        if fala is not None:
            dizer(fala, 'alertando' if partida.etapa == 5 else 'ensinando')
        else:
            dialogo = None
            partida.expressao = None
            if partida.etapa in (0, 5):
                concluir_licao()
            elif partida.etapa == 4:
                dizer('Aponte para selecionar Portão das Profundezas e feche a mão para confirmar.')
            else:
                dizer({1: 'Aponte para rolar até a marca.', 2: 'Abra a mão para saltar e pouse na plataforma.',
                       3: 'Aponte para a direita e ganhe velocidade na descida.'}[partida.etapa])

    def sair_do_tutorial():
        if not tutorial_avulso and save is not None and save.tutorial == len(ETAPAS_TREINO):
            iniciar_partida(continuar=True)
        else:
            titulo()

    opcoes_historia = ('Continuar', 'Novo Jogo', 'Rever história')

    def selecionar_inicio():
        nonlocal estado, opcao_inicio, bloqueio, tempo_gesto, creditos, tutorial_avulso
        if opcao_inicio == 1:
            tutorial_avulso = True
            abrir_tutorial()
        elif opcao_inicio == 2:
            estado, creditos = 'creditos', Creditos(arte_titulo)
            voz.interromper()
        elif save is None:
            iniciar_abertura()
        else:
            estado, opcao_inicio, bloqueio, tempo_gesto = 'historia', 0, True, 0.0

    def selecionar_historia():
        nonlocal estado, opcao_inicio, bloqueio, tempo_gesto
        opcao = opcoes_historia[opcao_inicio]
        if opcao == 'Continuar':
            iniciar_partida(continuar=True)
        elif opcao == 'Novo Jogo' and save is not None:
            estado, opcao_inicio, bloqueio, tempo_gesto = 'novo', 0, True, 0.0
        else:
            iniciar_abertura(rever=opcao == 'Rever história')

    def abrir_pausa(origem):
        nonlocal estado, origem_pausa, menu_indice, menu_ultimo_gesto, proximo_passo_menu, tempo_gesto, bloqueio
        origem_pausa, estado = origem, 'menu'
        menu_indice, menu_ultimo_gesto, proximo_passo_menu = 0, None, 0.0
        clique[0] = None
        tempo_gesto, bloqueio = 0.0, True

    if getattr(args, 'semente', None) is not None:
        iniciar_partida(args.semente)

    with ExitStack() as recursos:
        try:
            while True:
                agora = time.monotonic()
                decorrido = agora - anterior
                dt = min(decorrido, .1)
                anterior = agora
                if estado in ('jogo', 'fim', 'tutorial', 'menu'):
                    tempo_leitura = max(0.0, tempo_leitura - decorrido)
                if futuro is not None and futuro.done():
                    try:
                        controle = recursos.enter_context(futuro.result())
                    except Exception as erro:
                        erro_camera = str(erro)
                    futuro = None
                quadro = None
                leitura = None
                gesto = 'sem_mao'
                reconhecido = False
                eixo_analogico = None
                vetor_mapa = None
                passo_teclado = 0
                if controle is not None:
                    leitura, quadro = controle.ler()
                    if leitura.estado == 'falha_camera':
                        recursos.close()
                        controle = None
                        erro_camera = 'Camera parou de enviar imagem.'
                    elif punho_fechado(controle.pontos):
                        gesto = 'punho'
                        reconhecido = True
                    elif pose := gesto_da_mao(controle.pontos, aceitar_para_baixo=estado in ('mapa', 'inicio')):
                        gesto = pose
                        reconhecido = True
                        if pose in ('esquerda', 'direita', 'vertical'):
                            if estado == 'inicio':
                                gesto = direcao_menu_inicial(controle.pontos)
                            else:
                                eixo_analogico = eixo_indicador(controle.pontos)
                            if estado == 'mapa':
                                vetor = controle.pontos[8, :2] - controle.pontos[5, :2]
                                vetor_mapa = vetor / np.linalg.norm(vetor)
                    elif leitura.estado == 'reconhecido' and leitura.comando in NAVEGACAO:
                        gesto = leitura.comando
                        reconhecido = True
                if estado != 'tutorial' and (controle is None or estado in ('inicio', 'historia', 'novo', 'intro', 'menu', 'mapa', 'fim', 'gameover')):
                    eixo = eixo_teclado()
                    if estado == 'inicio' and not eixo:
                        tecla_pressionada = ctypes.windll.user32.GetAsyncKeyState
                        eixo = bool(tecla_pressionada(0x28) & 0x8000) - bool(tecla_pressionada(0x26) & 0x8000)
                    if eixo:
                        gesto = 'direita' if eixo > 0 else 'esquerda'
                        reconhecido = True
                        eixo_analogico = None
                        passo_teclado = eixo
                    elif estado == 'jogo':
                        if ctypes.windll.user32.GetAsyncKeyState(0x46) & 0x8000:
                            gesto = 'punho'
                            reconhecido = True
                if estado == 'tutorial':
                    tecla_pressionada = ctypes.windll.user32.GetAsyncKeyState
                    eixo = eixo_teclado()
                    if tecla_pressionada(0x46) & 0x8000:
                        gesto, reconhecido, eixo_analogico = 'punho', True, None
                    elif eixo:
                        gesto = 'direita' if eixo > 0 else 'esquerda'
                        reconhecido = True
                        eixo_analogico = eixo * (.5 if tecla_pressionada(0x10) & 0x8000 else 1)

                if gesto != gesto_candidato:
                    gesto_candidato = gesto
                    tempo_candidato = 0.0
                else:
                    tempo_candidato += dt
                if controle is not None and tempo_candidato < .12:
                    reconhecido = False

                estado_quadro = estado
                tela_mapa = estado == 'mapa'
                tela_intro = estado == 'intro'
                if estado in ('historia', 'novo', 'fim', 'gameover'):
                    desenhar_profundidade(tela, cavernas, None, 0)
                praticando = estado == 'tutorial' and dialogo is None and 1 <= partida.etapa <= 3
                gesto_liberacao = 'tres' if praticando else 'punho'
                if gesto != gesto_liberacao and estado != 'jogo':
                    bloqueio = False
                lado = NAVEGACAO.get(gesto, 0) if reconhecido else 0
                mudou_lado = lado and lado != ultimo_lado_inicio
                ultimo_lado_inicio = lado
                if estado in ('inicio', 'historia', 'novo', 'fim', 'gameover', 'intro', 'creditos'):
                    ativo = gesto == 'punho' and reconhecido and not bloqueio
                    tempo_gesto = tempo_gesto + dt if ativo else 0
                    confirmar = tempo_gesto >= .35 or menu_confirmar
                    if confirmar:
                        bloqueio, tempo_gesto = True, 0.0
                    menu_confirmar = False
                    if estado == 'inicio':
                        if mudou_lado:
                            opcao_inicio = (opcao_inicio + lado) % len(BOTOES)
                        desenhar_inicio(tela, arte_titulo, opcao_inicio)
                        if clique[0] is not None:
                            clicada = opcao_clicada(*clique[0])
                            clique[0] = None
                            if clicada is not None:
                                opcao_inicio, confirmar = clicada, True
                        if confirmar:
                            selecionar_inicio()
                    elif estado == 'creditos':
                        sair = creditos.atualizar(decorrido)
                        creditos.desenhar(tela)
                        if clique[0] is not None:
                            x, y = clique[0]
                            clique[0] = None
                            esquerda, topo, direita, base = SAIR_CREDITOS
                            sair |= esquerda <= x <= direita and topo <= y <= base
                        if confirmar or sair:
                            titulo()
                    elif estado == 'intro':
                        if mudou_lado:
                            intro.opcao = (intro.opcao + lado) % len(Introducao.BOTOES)
                        if confirmar:
                            intro.escolher(intro.opcao)
                        if clique[0] is not None:
                            intro.clicar(*clique[0])
                            clique[0] = None
                        terminou_intro = intro.atualizar(dt)
                        fala = intro.fala_do_mestre()
                        if fala:
                            voz.falar(fala)
                        intro.desenhar(tela, cavernas[0][ABERTURA[intro.pagina][1]])
                        if controle is not None and intro.saida is None:
                            tela.cartao((761, 12), (944, 108))
                            tela.texto('CNN', (773, 30), .34, espessura=1)
                            entrada = getattr(controle.classificador, 'imagem', None)
                            if entrada is not None:
                                tela.imagem(cv2.cvtColor(entrada, cv2.COLOR_GRAY2BGR),
                                            773, 38, 61, 61, cv2.INTER_NEAREST)
                            if quadro is not None:
                                tela.imagem(quadro, 840, 25, 96, 72)
                        if terminou_intro:
                            if rever_intro:
                                titulo()
                            else:
                                save = Progresso(random.randrange(2**32), tutorial_pulado=intro.pulada)
                                gravar()
                                if intro.pulada:
                                    iniciar_partida(continuar=True)
                                else:
                                    abrir_tutorial()
                    else:
                        if estado == 'historia':
                            manchete, opcoes = 'MODO HISTORIA', opcoes_historia
                        elif estado == 'novo':
                            manchete, opcoes = 'INICIAR NOVA JORNADA?', ('Voltar', 'Substituir o progresso salvo')
                        elif estado == 'fim':
                            manchete, opcoes = f'VITORIA | {partida.pontos} luzes', ('Voltar ao titulo',)
                        else:
                            manchete, opcoes = 'GAME OVER', ('Voltar ao titulo',)
                        if mudou_lado:
                            opcao_inicio = (opcao_inicio + lado) % len(opcoes)
                        linhas = [('> ' if i == opcao_inicio else '   ') + opcao for i, opcao in enumerate(opcoes)]
                        if estado == 'gameover':
                            volta_gameover -= decorrido
                            linhas += ['Seu progresso salvo fica em Play > Continuar.' if save else
                                       'Voltando ao titulo.']
                            confirmar |= volta_gameover <= 0
                        linhas += ['Indicador ou setas escolhem. Punho, Enter ou clique confirmam.']
                        if futuro is not None:
                            linhas.append('Preparando a camera...')
                        elif erro_camera:
                            linhas.append('Camera indisponivel. Use o teclado.')
                        painel(tela, manchete, linhas)
                        if clique[0] is not None:
                            x, y = clique[0]
                            clique[0] = None
                            if 145 <= x <= 800 and 225 <= y < 225 + 34 * len(opcoes):
                                opcao_inicio, confirmar = (y - 225) // 34, True
                        if confirmar and futuro is None:
                            if estado == 'historia':
                                selecionar_historia()
                            elif estado == 'novo' and opcao_inicio == 1:
                                iniciar_abertura()
                            else:
                                titulo()
                elif estado == 'mapa':
                    if vetor_mapa is None and gesto in NAVEGACAO:
                        vetor_mapa = (NAVEGACAO[gesto], 0)
                    mapa.atualizar(dt, vetor_mapa if reconhecido else None, partida.fase, passo_teclado)
                    parado = mapa.no == mapa.destino
                    ativo = parado and reconhecido and gesto == 'punho' and not bloqueio and futuro is None
                    tempo_gesto = tempo_gesto + dt if ativo else 0.0
                    desenhar_mapa(tela, partida, mapa, sprites_mapa, tempo_gesto / .35, futuro is not None,
                                  arte_mapa)
                    clicou_entrada = False
                    if clique[0] is not None:
                        x, y = clique[0]
                        clique[0] = None
                        clicou_entrada = 10 <= x <= 949 and 485 <= y <= 529
                    if parado and futuro is None and (tempo_gesto >= .35 or clicou_entrada):
                        entrar_fase()
                elif estado == 'tutorial':
                    if dialogo is not None or partida.etapa >= 4:
                        ativo = gesto == 'punho' and reconhecido and not bloqueio
                        tempo_gesto = tempo_gesto + dt if ativo else 0.0
                        confirmar = tempo_gesto >= .35 or menu_confirmar
                        if clique[0] is not None:
                            x, y = clique[0]
                            clique[0] = None
                            if dialogo is not None:
                                confirmar |= 191 <= x <= 770 and 486 <= y <= 528
                            elif partida.etapa == len(ETAPAS_TREINO):
                                confirmar |= 115 <= x <= 845 and 225 <= y <= 310
                            elif 230 <= x <= 730 and 369 <= y <= 449:
                                interacao, confirmar = int(y >= 409), True
                        if dialogo is None and partida.etapa == 4 and mudou_lado:
                            interacao = (interacao + lado) % 2
                        if confirmar:
                            bloqueio, tempo_gesto = True, 0.0
                            if dialogo is not None:
                                avancar_dialogo()
                            elif partida.etapa == 4:
                                if interacao == 1:
                                    concluir_licao()
                                else:
                                    dizer('Por ali voltamos ao vilarejo. Escolha o Portão das Profundezas.')
                            else:
                                sair_do_tutorial()
                    else:
                        tempo_gesto = tempo_gesto + dt if gesto == 'tres' and reconhecido and not bloqueio else 0.0
                        if tempo_gesto >= .7:
                            abrir_pausa('tutorial')
                        else:
                            acao = gesto if reconhecido else 'sem_mao'
                            eixo = eixo_analogico if reconhecido and eixo_analogico is not None else DIRECAO.get(acao, 0)
                            partida.atualizar(dt, eixo, acao == 'parar', controle is None, acao == 'sem_mao',
                                             frear=acao == 'punho', manter_embalo=acao == 'vertical')
                            if partida.objetivo_cumprido():
                                concluir_licao()
                    menu_confirmar = False
                    if estado in ('tutorial', 'menu'):
                        partida.desenhar(tela, sprites, cavernas)
                        if dialogo is not None:
                            if partida.etapa in (0, 5):
                                retrato = mestre[expressao_mestre]
                                x_mestre = partida.x - partida.camera_x + 115
                                colar(tela.pixels, retrato, round(x_mestre * tela.escala),
                                      round((partida.PISO - 66) * tela.escala))
                            tela.cartao((191, 486), (770, 528))
                            tela.texto('Avancar fala | Feche a mao, Enter ou clique aqui.', (209, 513), .46, espessura=1)
                        elif partida.etapa == 4:
                            tela.cartao((230, 349), (730, 449))
                            tela.texto(('> ' if interacao == 0 else '   ') + 'Voltar ao vilarejo', (249, 392), .54, espessura=1)
                            tela.texto(('> ' if interacao == 1 else '   ') + 'Portao das Profundezas', (249, 432), .54, espessura=1)
                elif estado == 'jogo':
                    fase_anterior = partida.fase
                    vidas_antes, inimigos_antes = partida.vidas, len(partida.inimigos)
                    checkpoint_anterior = partida.checkpoint
                    no_portal = bool(partida.tempo_vitoria)
                    if gesto != 'tres':
                        bloqueio = False
                    tempo_gesto = tempo_gesto + dt if gesto == 'tres' and reconhecido and not bloqueio else 0
                    if tempo_gesto >= .7:
                        abrir_pausa('jogo')
                    eixo = DIRECAO.get(gesto, 0) if reconhecido else 0
                    if reconhecido and eixo_analogico is not None:
                        eixo = eixo_analogico
                    if estado == 'jogo':
                        salto = reconhecido and gesto == 'parar'
                        partida.atualizar(dt, eixo, salto,
                                         quadro is None, gesto == 'sem_mao',
                                         frear=gesto == 'punho' and reconhecido,
                                         manter_embalo=eixo_analogico is not None and not eixo)
                        if partida.checkpoint != checkpoint_anterior and partida.fase == fase_anterior:
                            gravar_campanha()
                        if not partida.terminou and partida.fase == fase_anterior:
                            ocioso = ocioso + dt if abs(partida.vx) < 8 and partida.no_chao else 0.0
                            perigo = partida.perigo_proximo()
                            queda_repetida_pendente |= partida.alerta and partida.vidas < vidas_antes
                            if perigo is None:
                                perigo_anterior = None
                            if partida.tempo_vitoria and not no_portal:
                                dizer('Muito bem! Você encontrou o portal.', 'orgulhoso')
                            elif queda_repetida_pendente and (partida.tempo_fase >= proxima_fala_contextual or tempo_leitura <= 0):
                                dizer('Preda, meu filho... você é uma pedra, não um saco de batatas!', 'alertando')
                                queda_repetida_pendente = False
                                proxima_fala_contextual = partida.tempo_fase + 16
                            elif partida.vidas < vidas_antes and partida.tempo_fase >= proxima_fala_contextual:
                                dizer('Cuidado! Pegue impulso e observe o caminho antes de saltar.', 'alertando')
                                proxima_fala_contextual = partida.tempo_fase + 16
                            elif ocioso >= 12 and perigo is None and partida.tempo_fase >= proxima_fala_contextual and tempo_leitura <= 0:
                                dizer('Vai criar musgo aí, Preda? Temos uma montanha para salvar!')
                                proxima_fala_contextual = partida.tempo_fase + 28
                            elif partida.vidas > vidas_antes:
                                dizer('Boa! Você conquistou uma vida extra.', 'orgulhoso')
                            elif (partida.salto_dificil and partida.tempo_fase >= proxima_fala_contextual and tempo_leitura <= 0):
                                dizer('Muito bem! Se continuar assim, um dia vai deixar de ser uma pedrinha e virar um pedregulho!', 'orgulhoso')
                                proxima_fala_contextual = partida.tempo_fase + 20
                                partida.salto_dificil = False
                            elif len(partida.inimigos) < inimigos_antes:
                                dizer('Muito bem! Você derrotou a patrulha.', 'orgulhoso')
                            elif (perigo is not None and perigo != perigo_anterior
                                  and partida.tempo_fase >= proxima_fala_contextual and tempo_leitura <= 0):
                                perigo_anterior = perigo
                                dizer('Cuidado, garoto! Algumas rachaduras não são fáceis de consertar! ' + perigo[1], 'alertando')
                                proxima_fala_contextual = partida.tempo_fase + 15
                            elif not explicou_geiser and perigo is None and partida.perto_da_rampa():
                                if expressao_mestre == 'ensinando' or tempo_leitura <= 0:
                                    explicou_geiser = True
                                    dizer('Pule na rampa e atravesse o gêiser quando começar o jato.')
                            elif (expressao_mestre != 'ensinando' and tempo_leitura <= 0
                                  and perigo is None):
                                dizer('Observe o caminho e mantenha o embalo.', 'ensinando')
                    partida.desenhar(tela, sprites, cavernas)
                    proxima_dica -= decorrido
                    if not args.sem_treinador and partida.alerta and not partida.terminou:
                        resposta_durante = solicitar_dica(partida.dados())
                    if partida.alerta:
                        partida.alerta = False
                    if (not args.sem_treinador and not partida.terminou
                            and resposta_durante is None and proxima_dica <= 0):
                        resposta_durante = solicitar_dica(partida.dados())
                    if (not partida.terminou and tempo_leitura <= 0
                            and resposta_durante is not None and resposta_durante.done()):
                        dica, fonte = resposta_durante.result()
                        dizer(dica, 'alertando' if partida.vao_repetido else 'ensinando', fonte)
                        resposta_durante = None
                        proxima_dica = 20.0
                    if campanha is not None and (partida.venceu or partida.fase != fase_anterior):
                        completou_revisita = partida.fase != fase_anterior or partida.venceu
                        partida, campanha = campanha, None
                        estado, tempo_gesto, bloqueio = 'mapa', 0.0, True
                        clique[0] = None
                        if completou_revisita:
                            dizer('Boa! Você completou o percurso novamente.', 'orgulhoso')
                        else:
                            dizer('Cuidado com os perigos. Você pode tentar a fase de novo.', 'alertando')
                    elif partida.terminou:
                        estado = 'fim' if partida.venceu else 'gameover'
                        opcao_inicio = 0
                        bloqueio = True
                        tempo_gesto = 0
                        if partida.venceu:
                            gravar_campanha()
                            dizer('Parabéns! Você completou a expedição pelas cavernas.', 'orgulhoso')
                        else:
                            volta_gameover = 8.0
                            dizer('A trilha foi difícil. Observe os perigos na próxima tentativa.', 'alertando')
                        resposta = solicitar_dica(partida.dados()) if not args.sem_treinador else None
                    elif partida.fase != fase_anterior:
                        gravar_campanha()
                        estado, tempo_gesto, bloqueio = 'mapa', 0.0, True
                        clique[0] = None
                        dizer('Fase concluída! Aponte na direção da próxima entrada.', 'orgulhoso')
                elif estado == 'menu':
                    partida.desenhar(tela, sprites, cavernas)
                    if origem_pausa == 'tutorial':
                        opcoes_menu = ('Continuar', 'Voltar ao marco', 'Voltar ao titulo')
                    else:
                        opcoes_menu = ('Continuar', 'Voltar ao mapa', 'Voltar ao titulo')
                    direcao = NAVEGACAO.get(gesto, 0) if reconhecido else 0
                    if direcao and (direcao != menu_ultimo_gesto or agora >= proximo_passo_menu):
                        menu_indice = (menu_indice + direcao) % len(opcoes_menu)
                        proximo_passo_menu = agora + .85
                    if reconhecido or tempo_candidato >= .20:
                        menu_ultimo_gesto = direcao
                    linhas = [('> ' if i == menu_indice else '   ') + opcao
                              for i, opcao in enumerate(opcoes_menu)]
                    linhas += ['Indicador lateral: escolher   |   Punho: confirmar',
                               'Setas e Enter, ou clique na opcao.']
                    painel(tela, 'PAUSA', linhas)
                    ativo = reconhecido and gesto == 'punho' and not bloqueio
                    tempo_gesto = tempo_gesto + dt if ativo else 0
                    selecionar = menu_confirmar or tempo_gesto >= .35
                    menu_confirmar = False
                    if clique[0] is not None:
                        x, y = clique[0]
                        if 145 <= x <= 800 and 225 <= y < 225 + 34 * len(opcoes_menu):
                            menu_indice = (y - 225) // 34
                            selecionar = True
                        clique[0] = None
                    if selecionar:
                        if menu_indice == 0:
                            estado = origem_pausa
                            bloqueio = gesto == 'tres'
                            tempo_gesto = 0
                            if estado == 'tutorial' and dialogo is not None:
                                dizer(dialogo.atual)
                        elif menu_indice == 1:
                            if origem_pausa == 'tutorial':
                                abrir_tutorial(partida.etapa)
                            elif save is not None:
                                iniciar_partida(continuar=True)
                            else:
                                estado, tempo_gesto, bloqueio = 'mapa', 0.0, True
                        else:
                            titulo()
                if estado in ('fim', 'gameover') and tempo_leitura <= 0 and resposta is not None and resposta.done():
                    dica, fonte = resposta.result()
                    expressao_fim = 'orgulhoso' if partida.venceu else 'alertando' if partida.vao_repetido else 'ensinando'
                    dizer(dica, expressao_fim, fonte)
                    resposta = None
                if estado_quadro not in ('inicio', 'intro', 'creditos'):
                    desenhar_mestre(tela, mestre[expressao_mestre], dica_mestre, fonte_mestre,
                                     lateral=tela_mapa, voz_ativa=voz.ativa)
                    painel_camera(tela, quadro, controle, leitura, gesto, futuro is not None, lateral=tela_mapa)
                cv2.imshow(JANELA, tela.pixels)
                tecla = cv2.waitKey(1 if controle else 20) & 0xff
                if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if estado != estado_quadro:
                    clique[0] = None
                    continue
                if tela_intro:
                    if estado == 'intro':
                        if tecla in (27, ord('s')):
                            intro.escolher(Introducao.PULAR)
                        elif tecla == 32:
                            intro.escolher(1)
                        elif tecla == 13:
                            intro.escolher(intro.opcao)
                    continue
                if tecla == 27:
                    if estado == 'jogo' or estado == 'tutorial':
                        abrir_pausa(estado)
                    elif estado == 'inicio':
                        break
                    else:
                        titulo()
                    continue
                if clique[0] is not None:
                    x, y = clique[0]
                    esquerda, topo, direita, base = caixa_voz(tela_mapa)
                    if esquerda <= x <= direita and topo <= y <= base:
                        clique[0] = None
                        voz.ativar(not voz.ativa)
                        if voz.ativa:
                            voz.falar(dica_mestre)
                if estado in ('inicio', 'historia', 'novo', 'fim', 'gameover', 'menu', 'creditos') and tecla in (13, 32):
                    menu_confirmar = True
                elif estado == 'mapa' and tecla in (13, 32) and futuro is None and mapa.no == mapa.destino:
                    entrar_fase()
                elif estado == 'tutorial' and tecla == 13 and (dialogo is not None or partida.etapa >= 4):
                    menu_confirmar = True
                elif estado == 'tutorial' and dialogo is None and 1 <= partida.etapa <= 3 and tecla == ord('r'):
                    partida.reposicionar()
                elif estado in ('jogo', 'tutorial') and tecla == ord('m'):
                    abrir_pausa(estado)
                elif (estado == 'jogo' or estado == 'tutorial' and dialogo is None and partida.etapa < 4) and tecla == 32:
                    partida.pular()
        finally:
            voz.fechar()
            if futuro is not None:
                def fechar(f):
                    try:
                        f.result().__exit__(None, None, None)
                    except Exception:
                        pass
                futuro.add_done_callback(fechar)
            executor.shutdown(wait=False)
            cv2.destroyAllWindows()


class PartidaAventura:
    """Aventura de plataforma com movimento, salto e fases procedurais."""

    RAIO = 24
    PISO = 462
    VELOCIDADE_MAXIMA = 245
    VELOCIDADE_ATAQUE = 220
    ALTURA_INIMIGO = 35
    MEIA_LARGURA_INIMIGO = 20
    CICLO_GEISER = 3.6
    DURACAO_GEISER = .85
    AVISO_GEISER = .70

    def __init__(self, semente=None, fases=None):
        self.semente = random.randrange(2**32) if semente is None and fases is None else semente
        self.fases = gerar_expedicao(self.semente) if fases is None else fases
        self.quedas = {}
        self.alerta = False
        self.vao_repetido = False
        self.fase = 0
        self.pontos = 0
        self.vidas = 3
        self.sem_mao = self.teclado = 0.0
        self.trocas = self.direcao = 0
        self.terminou = False
        self.venceu = False
        self.salto_anterior = False
        self.vx = 0.0
        self.impacto = 0.0
        self.tempo_vitoria = 0.0
        self.carregar_fase()

    def carregar_fase(self):
        fase = self.fases[self.fase]
        self.x, self.y = 80.0, float(self.PISO - self.RAIO)
        self.vy = 0.0
        self.vx = 0.0
        self.no_chao = True
        self.camera_x = 0.0
        self.imune = 1.2
        self.luzes = list(fase["luzes"])
        self.vidas_extras = list(fase.get('vidas_extras', ()))
        self.inimigos = list(fase.get('inimigos', ()))
        self.jatos_usados = {}
        self.aviso_vida = 0.0
        self.salto_dificil = False
        self.checkpoint = 80.0
        self.angulo = 0.0
        self.tempo_fase = 0.0
        self.apoio = None
        self.vao_em_queda = None
        self.tolerancia_borda = self.fila_salto = 0.0

    def pular(self):
        if self.terminou or self.tempo_vitoria:
            return
        if self.no_chao or self.tolerancia_borda > 0:
            self.vy = -720.0
            self.no_chao = False
            self.apoio = None
            self.tolerancia_borda = self.fila_salto = 0.0
        else:
            self.fila_salto = .15

    def sofrer_dano(self):
        if self.imune > 0:
            return
        self.vidas -= 1
        self.impacto = 0.8
        if self.vidas <= 0:
            self.terminou = True
            return
        self.x = self.checkpoint
        self.y = self.PISO - self.RAIO
        self.vy = 0
        self.vx = 0
        self.no_chao = True
        self.apoio = None
        self.vao_em_queda = None
        self.tolerancia_borda = self.fila_salto = 0.0
        self.imune = 2.0

    def atualizar(self, dt, eixo, salto, teclado, sem_mao, frear=False, manter_embalo=False):
        if self.terminou:
            return
        if self.tempo_vitoria > 0:
            self.tempo_vitoria = max(0, self.tempo_vitoria - dt)
            if self.tempo_vitoria == 0:
                if self.fase + 1 == len(self.fases):
                    self.venceu = self.terminou = True
                else:
                    self.fase += 1
                    self.carregar_fase()
            return
        # Subpassos mantêm colisões estáveis mesmo quando a captura atrasa.
        self.sem_mao += dt if sem_mao and not teclado else 0
        self.teclado += dt if teclado else 0
        if eixo:
            direcao = -1 if eixo < 0 else 1
            if self.direcao == -direcao:
                self.trocas += 1
            self.direcao = direcao
        if salto and not self.salto_anterior:
            self.pular()
        self.salto_anterior = salto
        self.impacto = max(0, self.impacto - dt)
        self.aviso_vida = max(0, self.aviso_vida - dt)
        if eixo:
            alvo = eixo * self.VELOCIDADE_MAXIMA
            passo_velocidade = 600 * dt
            self.vx += max(-passo_velocidade, min(passo_velocidade, alvo - self.vx))
        elif frear or (self.no_chao and not salto and not manter_embalo):
            desaceleracao = (450 if frear else 55) * dt
            self.vx -= math.copysign(min(abs(self.vx), desaceleracao), self.vx)
        passos = max(1, math.ceil(dt / (1 / 120)))
        for _ in range(passos):
            self.avancar(dt / passos, eixo)
            if self.terminou or self.tempo_vitoria:
                break

    def avancar(self, dt, eixo):
        fase = self.fases[self.fase]
        estava_no_chao = self.no_chao
        apoio_anterior = self.apoio
        self.fila_salto = max(0.0, self.fila_salto - dt)
        self.tolerancia_borda = max(0.0, self.tolerancia_borda - dt)
        movel_anterior = self.plataforma_movel()
        self.tempo_fase += dt
        movel = self.plataforma_movel()
        if self.apoio == 'movel':
            self.x += movel[0] - movel_anterior[0]
        self.apoio = None
        anterior_x, anterior_y = self.x, self.y
        self.x = max(self.RAIO, min(fase["chao"][-1][1] - self.RAIO, self.x + self.vx * dt))
        self.angulo += self.vx * dt / self.RAIO
        self.vy = min(self.vy + 1550 * dt, 1000)
        self.y += self.vy * dt
        self.no_chao = False
        superficies = [(inicio, fim, self.PISO) for inicio, fim in fase["chao"]]
        superficies += fase["plataformas"]
        if movel:
            superficies.append(movel)
        if self.vy >= 0:
            for inicio, fim, altura in superficies:
                if (inicio < self.x + self.RAIO and self.x - self.RAIO < fim
                        and anterior_y + self.RAIO <= altura <= self.y + self.RAIO):
                    self.y = altura - self.RAIO
                    self.vy = 0
                    self.no_chao = True
                    self.vao_em_queda = None
                    if (inicio, fim, altura) == movel:
                        self.apoio = 'movel'
                    break
        self.passar_pelas_rampas(fase.get('rampas', ()), anterior_x, anterior_y, apoio_anterior)
        if self.no_chao and not estava_no_chao:
            self.salto_dificil |= any(i in self.jatos_usados and inicio <= self.x <= fim and
                                      abs(self.y + self.RAIO - altura) < 1
                                      for i, (inicio, fim, altura) in enumerate(fase.get('patamares_bonus', ())))
        if self.no_chao and self.fila_salto > 0:
            self.pular()
        if estava_no_chao and not self.no_chao and self.vy > 0:
            self.tolerancia_borda = .12
        self.imune = max(0, self.imune - dt)
        if self.y > self.PISO - self.RAIO and self.vao_em_queda is None:
            for i, ((_, borda), (chegada, _)) in enumerate(zip(fase['chao'], fase['chao'][1:])):
                if borda - self.RAIO <= self.x <= chegada + self.RAIO:
                    self.vao_em_queda = (self.fase, i)
                    break
        if self.y > ALTURA + 50:
            if (chave := self.vao_em_queda) is not None and self.imune <= 0:
                self.quedas[chave] = self.quedas.get(chave, 0) + 1
                if self.quedas[chave] == 2:
                    self.alerta = self.vao_repetido = True
            self.sofrer_dano()
            return
        if self.y + self.RAIO >= self.PISO - 15:
            for inicio, fim in fase["espinhos"]:
                if inicio < self.x + self.RAIO and self.x - self.RAIO < fim:
                    self.sofrer_dano()
                    return
        if self.colidir_inimigos(anterior_y):
            return
        for luz in self.luzes[:]:
            if math.hypot(self.x - luz[0], self.y - luz[1]) < 38:
                self.luzes.remove(luz)
                self.pontos += 1
        for vida in self.vidas_extras[:]:
            if math.hypot(self.x - vida[0], self.y - vida[1]) <= 33:
                self.vidas_extras.remove(vida)
                self.vidas += 1
                self.aviso_vida = 2.0
        if self.no_chao and self.y == self.PISO - self.RAIO:
            for ponto in fase.get('checkpoints', ()):
                if self.checkpoint < ponto <= self.x <= ponto + 66:
                    self.checkpoint = ponto
        self.camera_x = max(0, min(self.x - 320, fase["chao"][-1][1] - LARGURA))
        if fase.get('portal', True) and self.x >= fase["chao"][-1][1] - 80 and self.no_chao:
            self.tempo_vitoria = 1.5
            self.vx = 0

    def estado_geiser(self, fase_ciclo):
        ciclo, fase = divmod(self.tempo_fase + fase_ciclo, self.CICLO_GEISER)
        return int(ciclo), fase

    def passar_pelas_rampas(self, rampas, anterior_x, anterior_y, apoio_anterior):
        for i, (inicio, fim, base, topo, fase_ciclo) in enumerate(rampas):
            inclinacao = (base - topo) / (fim - inicio)
            subindo = apoio_anterior == ('rampa', i)
            altura = base - (self.x - inicio) * inclinacao
            altura_anterior = base - (anterior_x - inicio) * inclinacao
            cruzou = (anterior_y + self.RAIO <= altura_anterior + 1
                      and self.y + self.RAIO >= altura and self.vy >= 0)
            if inicio <= self.x < fim and (subindo or cruzou):
                self.y, self.vy = altura - self.RAIO, 0.0
                self.no_chao, self.apoio = True, ('rampa', i)
                self.vao_em_queda = None
            elif subindo and self.x >= fim:
                self.vy = -max(0, self.vx) * inclinacao
            if fase_ciclo is None or abs(self.x - fim) > 23 or not topo - 96 <= self.y <= topo - 13:
                continue
            ciclo, fase = self.estado_geiser(fase_ciclo)
            if fase < self.DURACAO_GEISER and self.jatos_usados.get(i) != ciclo:
                self.jatos_usados[i] = ciclo
                self.vy = min(self.vy, -835.0)
                self.no_chao, self.apoio = False, None
                self.tolerancia_borda = self.fila_salto = 0.0

    def patrulha(self, inimigo):
        esquerda, direita, velocidade, offset = inimigo
        largura = direita - esquerda
        passo = (self.tempo_fase * velocidade + offset) % (2 * largura)
        return esquerda + min(passo, 2 * largura - passo), 1 if passo < largura else -1

    def perto_da_rampa(self):
        return any(inicio - 240 <= self.x <= fim + 70
                   for inicio, fim, *_ in self.fases[self.fase].get('rampas', ()))

    def perigo_proximo(self):
        fase = self.fases[self.fase]
        direcao = -1 if self.vx < -20 else 1
        for inimigo in self.inimigos:
            x, _ = self.patrulha(inimigo)
            if -35 <= (x - self.x) * direcao <= 230:
                return inimigo, 'Cuidado com a patrulha! Ganhe velocidade ou caia por cima do inimigo.'
        for inicio, fim in fase['espinhos']:
            borda = inicio if direcao > 0 else fim
            if 0 <= (borda - self.x) * direcao <= 205:
                return (inicio, fim), 'Espinhos à frente! Salte antes de encostar neles.'
        for (_, inicio), (fim, _) in zip(fase['chao'], fase['chao'][1:]):
            borda = inicio if direcao > 0 else fim
            if 0 <= (borda - self.x) * direcao <= 170:
                return (inicio, fim), 'Cuidado com o vão! Pegue embalo e salte perto da borda.'
        return None

    def colidir_inimigos(self, anterior_y):
        altura = self.PISO - self.ALTURA_INIMIGO
        for inimigo in self.inimigos[:]:
            x, _ = self.patrulha(inimigo)
            dx = abs(self.x - x)
            largura = self.MEIA_LARGURA_INIMIGO
            if dx > largura + self.RAIO or self.y + self.RAIO < altura:
                continue
            if anterior_y + self.RAIO <= altura and self.vy > 0 and dx <= largura + 9:
                self.inimigos.remove(inimigo)
                self.y, self.vy = altura - self.RAIO, -455.0
                self.no_chao, self.apoio = False, None
                self.tolerancia_borda = self.fila_salto = 0.0
            elif math.hypot(max(0, dx - largura), max(0, altura - self.y)) <= self.RAIO:
                if abs(self.vx) >= self.VELOCIDADE_ATAQUE and self.y >= altura - 8:
                    self.inimigos.remove(inimigo)
                elif self.imune <= 0:
                    self.sofrer_dano()
                    return True
        return False

    def dados(self):
        return {"pontos": self.pontos, "segundos_sem_mao": round(self.sem_mao, 1),
                "segundos_teclado": round(self.teclado, 1), "trocas_de_direcao": self.trocas,
                "vao_repetido": self.vao_repetido}

    def plataforma_movel(self):
        dados = self.fases[self.fase].get('movel')
        if dados is None:
            return None
        x, altura, amplitude = dados
        esquerda = x + round(amplitude * math.sin(self.tempo_fase * 1.4))
        return esquerda, esquerda + 80, altura

    def desenhar(self, tela, sprites, cavernas):
        fase = self.fases[self.fase]
        ambiente = self.fase
        desenhar_profundidade(tela, cavernas, fase.get('fundo', ambiente), self.camera_x)
        deslocamento = int(self.camera_x)
        luz_ambiente = AMBIENTES[ambiente][3]
        for x, y, tamanho in fase.get('decoracao', ()):
            x -= int(self.camera_x * .85)
            if -40 < x < LARGURA + 40:
                oscilacao = math.sin(self.tempo_fase * 1.3 + x * .04)
                if oscilacao > -.35:
                    tela.circulo((x, y - 25 + oscilacao * 8), tamanho / 13, luz_ambiente, -1)
        for inicio, fim in fase["chao"]:
            desenhar_rocha(tela, inicio, fim, self.PISO, ALTURA, deslocamento, fase.get('tom', 0), ambiente)
        for inicio, fim, altura in fase["plataformas"]:
            desenhar_rocha(tela, inicio, fim, altura, altura + 20, deslocamento, fase.get('tom', 0), ambiente)
        if movel := self.plataforma_movel():
            inicio, fim, altura = movel
            desenhar_rocha(tela, inicio, fim, altura, altura + 20, deslocamento, ambiente=ambiente)
            tela.linha((inicio - deslocamento, altura), (fim - deslocamento, altura), (189, 181, 110), 2)
        for ponto in fase.get('checkpoints', ()):
            x = ponto - deslocamento
            if -20 <= x <= LARGURA + 20:
                cor = (120, 220, 160) if ponto <= self.checkpoint else (138, 131, 143)
                tela.linha((x, self.PISO), (x, self.PISO - 52), (142, 145, 164), 2)
                tela.poligono([(x, self.PISO - 52), (x + 23, self.PISO - 46),
                               (x, self.PISO - 36)], cor)
        for inicio, fim, base, topo, fase_ciclo in fase.get('rampas', ()):
            x = fim - deslocamento
            if -48 <= x and inicio - deslocamento <= LARGURA + 30:
                desenhar_rampa(tela, inicio - deslocamento, x, base, topo, ambiente)
                if fase_ciclo is not None:
                    _, fase_jato = self.estado_geiser(fase_ciclo)
                    desenhar_geiser(tela, x, topo, fase_jato, self.CICLO_GEISER,
                                     self.DURACAO_GEISER, self.AVISO_GEISER, ambiente)
        for inicio, fim in fase["espinhos"]:
            for x in range(inicio, fim, 20):
                pontos = np.array([[x - deslocamento, self.PISO], [x + 10 - deslocamento, self.PISO - 22], [x + 20 - deslocamento, self.PISO]], np.int32)
                tela.poligono(pontos, (96, 91, 211))
                tela.poligono([pontos[0], pontos[1], (x + 10 - deslocamento, self.PISO)], (145, 145, 250))
        for x, y in self.luzes:
            if -20 < x - deslocamento < LARGURA + 20:
                x -= deslocamento
                tela.circulo((x, y), 16, (110, 220, 255), 1)
                topo, base = (x, y - 12), (x, y + 12)
                tela.poligono([topo, (x - 8, y - 3), base, (x + 9, y - 3)], (77, 170, 235))
                tela.poligono([topo, (x - 8, y - 3), base, (x + 2, y - 4)], (164, 242, 255))
        for x, y in self.vidas_extras:
            if -25 < x - deslocamento < LARGURA + 25:
                desenhar_vida(tela, x - deslocamento, y, self.tempo_fase)
        for inimigo in self.inimigos:
            x, direcao = self.patrulha(inimigo)
            if -30 < x - deslocamento < LARGURA + 30:
                desenhar_inimigo(tela, x - deslocamento, self.PISO, direcao, self.tempo_fase)
        if fase.get('portal', True):
            portal = fase["chao"][-1][1] - 60 - deslocamento
            tela.elipse((int(portal), self.PISO - 53), (28, 53), 0, 0, 360, (230, 210, 120), 5, cv2.LINE_AA)
        cx, cy = int(self.x - deslocamento), int(self.y)
        if getattr(self, 'expressao', None):
            expressao = self.expressao
        elif self.tempo_vitoria:
            expressao = 'vitoria'
        elif self.impacto:
            expressao = 'impacto'
        elif not self.no_chao:
            expressao = 'susto-queda' if self.vy > 100 else 'pulando'
        elif abs(self.vx) > 220:
            expressao = 'velocidade' if abs(self.vx) >= 245 else 'rolando'
        elif self.vx:
            expressao = 'rolando'
        else:
            expressao = 'parado'
        imagem = sprites[expressao]
        if expressao in ('rolando', 'velocidade'):
            lado = imagem.shape[0]
            matriz = cv2.getRotationMatrix2D((lado / 2, lado / 2), -math.degrees(self.angulo) % 360, 1)
            imagem = cv2.warpAffine(imagem, matriz, (lado, lado), flags=cv2.INTER_LINEAR)
        colar(tela.pixels, imagem, (cx - 32) * tela.escala, (cy - 32) * tela.escala)
        if not fase.get('portal', True):
            return
        tela.cartao((10, 8), (382, 41))
        tela.texto(f'Fase {self.fase + 1}/{len(self.fases)}  |  Luzes {self.pontos}  |  Vidas {self.vidas}', (20, 31), .59)
        if self.semente is not None:
            tela.texto(f'Semente {self.semente}', (402, 31), .42, espessura=1)
        if self.aviso_vida:
            tela.texto('+1 VIDA', (490, 31), .56, (153, 177, 255), 2)
        tela.cartao((10, 490), (541, 528))
        tela.texto('Indicador: velocidade  |  Palma: pular  |  Punho: frear  |  3 dedos: menu',
                   (20, 515), .37, espessura=1)
        tela.cartao((560, 490), (949, 528))
        tela.texto(AMBIENTES[ambiente][0], (577, 506), .45, AMBIENTES[ambiente][3], 1)
        instrucao = 'Salte na rampa e acerte o jato.' if self.perto_da_rampa() else 'Inimigo: embale ou caia por cima.'
        tela.texto(instrucao, (577, 520), .35, espessura=1)
        velocidade = abs(self.vx)
        cor = (106, 223, 255) if velocidade >= self.VELOCIDADE_ATAQUE else (144, 149, 176)
        tela.texto('ATAQUE' if velocidade >= self.VELOCIDADE_ATAQUE else 'VELOCIDADE',
                   (846, 507), .31, cor, 1)
        tela.retangulo((846, 512), (927, 519), (77, 67, 79))
        tela.retangulo((846, 512), (846 + 81 * velocidade / self.VELOCIDADE_MAXIMA, 519), cor)
        marca = 846 + 81 * self.VELOCIDADE_ATAQUE / self.VELOCIDADE_MAXIMA
        tela.linha((marca, 511), (marca, 520), (189, 226, 248))


class Treino(PartidaAventura):
    """Fase 00, com a física da aventura e marcos restauráveis."""

    MARCOS = (440, 440, 710, 1030, 1340, 1340, 1340)
    RAMPA = (1100, 1320, 397, PartidaAventura.PISO, None)

    def __init__(self, etapa=0):
        fase = dict(chao=[(0, 1800)], plataformas=[(930, 1100, 397), (1430, 1550, 390)],
                    espinhos=[], luzes=[], rampas=[self.RAMPA], fundo='tutorial', portal=False)
        super().__init__(fases=(fase,))
        self.etapa = etapa
        self.expressao = None
        self.reposicionar()

    def reposicionar(self):
        self.x = self.checkpoint = self.MARCOS[self.etapa]
        self.y = (397 if self.etapa == 3 else self.PISO) - self.RAIO
        self.camera_x = max(0, min(self.x - 320, 1800 - LARGURA))
        self.vx = self.vy = 0
        self.no_chao, self.apoio = True, None
        self.tolerancia_borda = self.fila_salto = 0

    def objetivo_cumprido(self):
        if self.etapa == 1:
            return 750 <= self.x <= 817 and self.no_chao
        if self.etapa == 2:
            return 930 <= self.x <= 1100 and self.no_chao and abs(self.y + self.RAIO - 397) < 1
        if self.etapa == 3:
            inicio, fim, *_ = self.RAMPA
            return inicio + 75 <= self.x <= fim and self.apoio == ('rampa', 0) and self.vx >= 180
        return False

    def desenhar(self, tela, sprites, cavernas):
        super().desenhar(tela, sprites, cavernas)
        if self.etapa in (1, 2, 3):
            x, y = ((783, self.PISO), (1015, 397), (1240, 438))[self.etapa - 1]
            x -= self.camera_x
            if 0 < x < LARGURA:
                cor = (128, 223, 255)
                tela.elipse((x, y - 2), (20, 6), 0, 0, 360, cor, 2)
                tela.poligono([(x, y - 11), (x - 8, y - 24), (x + 8, y - 24)], cor)
        tela.cartao((10, 7), (583, 42))
        tela.texto('FASE 00 | O BERCO DAS PEDRAS', (22, 30), .51, (189, 226, 248), 1)
        tela.retangulo((351, 19), (521, 30), (73, 60, 74))
        if self.etapa:
            tela.retangulo((351, 19), (351 + 170 * self.etapa / len(ETAPAS_TREINO), 30), (115, 208, 165))
        tela.texto(f'{self.etapa}/{len(ETAPAS_TREINO)}', (538, 29), .39, espessura=1)
        if self.etapa == len(ETAPAS_TREINO):
            tela.cartao((115, 150), (845, 320))
            texto_acentuado(tela, 'TUTORIAL CONCLUÍDO!', 145, 173, 33, '#ffe2bd', True)
            texto_acentuado(tela, 'SUA JORNADA COMEÇA AGORA!', 145, 228, 21, '#f6eedf', True)
            mensagem = ('Punho, Enter ou clique para voltar ao título.' if getattr(self, 'avulso', False) else
                        'Fase 01 liberada. Punho, Enter ou clique para ir ao mapa.')
            texto_acentuado(tela, mensagem, 145, 275, 17, '#f6eedf')
        else:
            tela.cartao((10, 486), (949, 528))
            tela.texto(sem_acentos(ETAPAS_TREINO[self.etapa]), (23, 512), .54, espessura=1)
            dicas = {1: 'Indicador ou setas: role ate a marca.',
                     2: 'Palma aberta ou Espaco: pule e pouse na plataforma.',
                     3: 'Indicador lateral ou seta direita: acelere na descida.'}
            if self.etapa in dicas:
                tela.texto(dicas[self.etapa], (436, 511), .40, espessura=1)


def validar_percurso(fase, antecedencia):
    """Percorre a rota principal no mesmo simulador usado durante o jogo."""
    partida = PartidaAventura(fases=(fase,))
    bordas = sorted([fim for _, fim in fase['chao'][:-1]] +
                    [inicio for inicio, _ in fase['espinhos']])
    obstaculos = iter(bordas)
    trecho = next(obstaculos, None)
    limite = math.ceil((fase['chao'][-1][1] / partida.VELOCIDADE_MAXIMA + 4) * 120)
    for _ in range(limite):
        saltar = trecho is not None and partida.x >= trecho - antecedencia
        if saltar:
            trecho = next(obstaculos, None)
        partida.atualizar(1 / 120, 0 if saltar else 1, saltar, False, False)
        if partida.vidas < 3:
            return False
        if partida.tempo_vitoria:
            return trecho is None
    return False


def gerar_expedicao(semente):
    rng = random.Random(semente)
    fases = []
    for etapa in range(len(ETAPAS)):
        for _ in range(24):
            chao = []
            x = 0
            total_trechos = 8 + etapa
            trechos_rampa = (2,) if etapa < 2 else (2, total_trechos - 3)
            for i in range(total_trechos):
                minimo = 510 if i in trechos_rampa or i == total_trechos - 1 else 405
                fim = x + rng.randrange(minimo, minimo + 91)
                chao.append((x, fim))
                x = fim + rng.randrange(64 + etapa * 8, 99 + etapa * 8)
            trecho_desvio = rng.choice([i for i in range(1, total_trechos - 1) if i not in trechos_rampa])
            inicio_desvio = chao[trecho_desvio][0] + rng.randrange(58, 83)
            altura = rng.randrange(354, 387)
            largura = rng.randrange(90, 120)
            plataformas = [(inicio_desvio, inicio_desvio + largura, altura)]
            movel = (inicio_desvio + largura + 6, altura - 46, rng.randrange(15, 26))
            luzes = [(inicio + rng.randrange(65, 108), 416) for inicio, _ in chao]
            luzes += [(inicio_desvio + largura // 2, altura - 45), (movel[0] + 40, movel[1] - 45)]
            espinhos = []
            if etapa:
                x_espinhos = chao[-1][0] + rng.randrange(230, 261)
                espinhos.append((x_espinhos, x_espinhos + 20 * (2 + etapa // 2)))
            rampas, vidas_extras = [], []
            for i in trechos_rampa:
                inicio = chao[i][0] + rng.randrange(150, 175)
                fim = inicio + rng.randrange(125, 145)
                rampas.append((inicio, fim, 407, 365, rng.uniform(0, PartidaAventura.CICLO_GEISER)))
                plataformas.append((fim + 126, fim + 250, 193))
                vidas_extras.append((fim + 182, 160))
                luzes.extend(((fim + 142, 158), (fim + 226, 160)))
            inimigos = []
            for i in range(1, total_trechos - 1):
                if i in trechos_rampa or (i + etapa) % 3 == 0:
                    continue
                inicio, fim = chao[i]
                esquerda, direita = inicio + 166, fim - 130
                inimigos.append((esquerda, direita, rng.randrange(42 + etapa * 8, 67 + etapa * 8),
                                 rng.randrange(2 * (direita - esquerda))))
            checkpoints = [chao[i][0] + 70 for i in (total_trechos // 3, 2 * total_trechos // 3)]
            decoracao = []
            for i, x in enumerate(range(90, chao[-1][1], 82)):
                camada, tamanho, deslocamento = rng.randrange(168, 358), rng.randrange(6, 23), rng.randrange(3)
                if i % 2 == 0:
                    decoracao.append((x + 9 * deslocamento, 390 + camada % 68, tamanho + 4))
            fase = dict(chao=chao, plataformas=plataformas, espinhos=espinhos, luzes=luzes,
                        movel=movel, decoracao=decoracao, tom=rng.randrange(-10, 15),
                        rampas=rampas, vidas_extras=vidas_extras, inimigos=inimigos, checkpoints=checkpoints,
                        patamares_bonus=plataformas[1:])
            if all(validar_percurso(fase, margem) for margem in (42, 56, 70)):
                fases.append(fase)
                break
        else:
            raise RuntimeError('Não foi possível montar uma rota de expedição válida.')
    return tuple(fases)
