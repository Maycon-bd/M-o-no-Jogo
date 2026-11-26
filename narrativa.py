"""Texto, progressão de diálogos e apresentação da abertura de Pedra Perdida."""
from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ABERTURA = (
    ('O NASCIMENTO', 'tutorial', (
        'Muito abaixo da superfície, onde a luz do sol jamais alcançou, existe um mundo esquecido pelo tempo.',
        'Um mundo onde as pedras possuem vida, sentimentos e histórias.',
        'No coração desse reino, um pequeno vilarejo guarda um segredo: o Berço das Pedras, lugar onde toda nova vida começa.',
    )),
    ('O SILÊNCIO DA MONTANHA', 4, (
        'Durante milhares de anos, o Coração da Montanha manteve esse mundo em equilíbrio.',
        'Até que, em uma noite, seu poderoso pulsar começou a desaparecer.',
        'As cavernas tremeram. Cristais perderam o brilho. Rachaduras surgiram onde antes havia vida.',
        'E, pela primeira vez, o Berço das Pedras ficou em silêncio.',
    )),
    ('UMA PEDRA DIFERENTE', 'tutorial', (
        'Mas, quando todos acreditavam que nenhuma outra pedra nasceria, um último tremor percorreu o vilarejo.',
        'De dentro do Berço surgiu uma pequena pedra cinzenta.',
        'Seu nome era Preda.',
        'Sem poderes extraordinários. Sem aparência imponente.',
        'Apenas um pequeno herói que ainda não conhecia seu destino.',
    )),
    ('O CONSELHO DO MESTRE', 5, (
        'Mestre Granito, a pedra mais antiga do vilarejo, percebeu algo que ninguém mais havia notado.',
        'Preda conseguia sentir um eco vindo das profundezas.',
        'Um chamado fraco, distante, como se a própria montanha estivesse pedindo ajuda.',
        '— Preda, meu garoto, talvez você não saiba de onde veio. Mas acredito que chegou a hora de descobrir até onde pode ir.',
    )),
    ('A JORNADA COMEÇA', 7, (
        'Para restaurar o Coração da Montanha, será preciso atravessar cavernas esquecidas, rios de magma, lagos subterrâneos e abismos onde nenhuma pedra ousou rolar.',
        'O destino de um mundo inteiro está nas mãos de alguém que sequer possui mãos.',
        'E assim começa a jornada de Preda...',
        'A Pedra Perdida.',
    )),
)

ETAPAS_TREINO = ('O Despertar', 'Movimentação', 'Pulo', 'Velocidade', 'Interação', 'Primeira Missão')
DESPERTAR = (
    'Ah! Então você finalmente acordou!',
    'Seja bem-vindo ao mundo, Preda!',
    'Eu sou Mestre Granito. E, pelo visto, vou ter bastante trabalho com você!',
    'Vamos começar pelo básico. Primeiro precisamos descobrir se você sabe rolar!',
)
ROLAR = ('Vamos lá, rapaz! Role até aquele ponto marcado.',)
PULAR = (
    'Muito bem! Pelo menos você não nasceu quadrado!',
    'Agora vamos descobrir se você consegue tirar os pés do chão!',
    'Ah, é verdade... você não tem pés.',
    'Bom, apenas pule!',
)
VELOCIDADE = (
    'Excelente! Já está quase parecendo uma pedra de verdade!',
    'Uma pedra parada não vai muito longe, Preda!',
    'Use aquela descida para ganhar velocidade!',
)
INTERACAO = (
    'Agora sim! Só não vá rolar para fora do vilarejo!',
    'Nem tudo se resolve rolando, meu garoto!',
    'Às vezes precisamos parar, observar e escolher o caminho.',
)
MISSAO = (
    'Ha! Nada mal para alguém que acabou de nascer!',
    'Mas agora preciso lhe contar uma coisa...',
    'Existe algo errado nas profundezas da montanha.',
    'Os cristais estão perdendo sua luz. As cavernas estão mudando.',
    'E há muitos anos eu não sentia um eco como o seu.',
    'Talvez você seja capaz de descobrir o que está acontecendo.',
    'Vá até o Portão das Profundezas.',
    'E lembre-se, meu garoto...',
    'Até a menor das pedras pode mover uma montanha!',
)
FALAS_TREINO = (DESPERTAR, ROLAR, PULAR, VELOCIDADE, INTERACAO, MISSAO)


class Dialogo:
    def __init__(self, falas):
        self.falas = iter(falas)
        self.atual = next(self.falas, None)

    def avancar(self):
        self.atual = next(self.falas, None)
        return self.atual


@lru_cache(maxsize=12)
def fonte(tamanho, destaque=False):
    nome = 'segoeuib.ttf' if destaque else 'segoeui.ttf'
    caminho = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / nome
    try:
        return ImageFont.truetype(str(caminho), tamanho)
    except OSError:
        return ImageFont.load_default(size=tamanho)


def quebrar(paragrafo, letra, largura):
    linhas, linha = [], ''
    for palavra in paragrafo.split():
        candidata = f'{linha} {palavra}' if linha else palavra
        if linha and letra.getlength(candidata) > largura:
            linhas.append(linha)
            linha = palavra
        else:
            linha = candidata
    return linhas + [linha]


def texto_acentuado(tela, texto, x, y, tamanho, cor, destaque=False):
    x, y = tela.ponto((x, y))
    letra = fonte(tamanho * tela.escala, destaque)
    largura = min(tela.pixels.shape[1] - x, int(letra.getlength(texto)) + tela.escala * 4)
    altura = min(tela.pixels.shape[0] - y, tamanho * tela.escala * 2)
    janela = tela.pixels[y:y + altura, x:x + largura]
    imagem = Image.fromarray(cv2.cvtColor(janela, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(imagem).text((0, 0), texto, font=letra, fill=cor)
    janela[:] = cv2.cvtColor(np.asarray(imagem), cv2.COLOR_RGB2BGR)


class Introducao:
    BOTOES = ('Voltar', 'Revelar texto', 'Avançar', 'Pular e ir ao mapa')
    PULAR = 3
    INICIO_BOTOES = 50
    LARGURA_BOTAO = 215
    DURACAO_FADE = .34

    def __init__(self, rever=False):
        self.pagina = 0
        self.opcao = 1
        self.caracteres = 0.0
        self.entrada = self.DURACAO_FADE
        self.saida = None
        self.destino = None
        self.fala_tocada = False
        self.pulada = False
        self.rever = rever

    def atualizar(self, dt):
        if self.saida is not None:
            self.saida += dt
            if self.saida >= self.DURACAO_FADE:
                if self.destino == len(ABERTURA):
                    return True
                self.pagina, self.saida = self.destino, None
                self.entrada, self.caracteres, self.opcao = self.DURACAO_FADE, 0.0, 1
                self.fala_tocada = False
        else:
            self.entrada = max(0, self.entrada - dt)
            self.caracteres += dt * 48
        return False

    def escolher(self, opcao):
        if self.saida is not None:
            return
        if opcao == 1 or (opcao == 2 and self.caracteres < self.tamanho_texto):
            self.caracteres = self.tamanho_texto
            self.opcao = 2
        elif opcao == self.PULAR:
            self.pulada = True
            self.destino, self.saida = len(ABERTURA), 0.0
        elif opcao != 0 or self.pagina:
            self.destino, self.saida = self.pagina + (1 if opcao == 2 else -1), 0.0

    @property
    def tamanho_texto(self):
        return sum(len(p) + 1 for p in ABERTURA[self.pagina][2])

    def clicar(self, x, y):
        indice = int((x - self.INICIO_BOTOES) // self.LARGURA_BOTAO)
        if (453 <= y <= 491 and 0 <= indice < len(self.BOTOES)
                and x - self.INICIO_BOTOES - indice * self.LARGURA_BOTAO <= 194):
            self.escolher(indice)

    def fala_do_mestre(self):
        if self.pagina == 3 and self.saida is None and not self.fala_tocada:
            paragrafos = ABERTURA[self.pagina][2]
            if self.caracteres > sum(len(p) + 1 for p in paragrafos[:-1]):
                self.fala_tocada = True
                return paragrafos[-1].removeprefix('— ')

    def desenhar(self, tela, caminho):
        h, w = tela.pixels.shape[:2]
        arte = quadro_abertura(caminho, w, h)
        tela.pixels[:] = arte
        escala = tela.escala
        area = Image.fromarray(cv2.cvtColor(tela.pixels, cv2.COLOR_BGR2RGB))
        desenho = ImageDraw.Draw(area)
        titulo, _, paragrafos = ABERTURA[self.pagina]
        def escrever(x, y, texto, tamanho, cor, destaque=False):
            desenho.text((x * escala, y * escala), texto, fill=cor,
                          font=fonte(tamanho * escala, destaque))
        escrever(50, 41, 'PEDRA PERDIDA', 16, '#ffe0ad', True)
        escrever(50, 73, titulo, 31, '#fff1d8', True)
        y, restantes = 133, int(self.caracteres)
        letra = fonte(21 * escala)
        for paragrafo in paragrafos:
            restante_paragrafo = paragrafo
            for linha in quebrar(paragrafo, letra, 830 * escala):
                restante_paragrafo = restante_paragrafo.lstrip()
                if restantes > 0:
                    escrever(50, y, linha[:restantes], 21, '#f7f0eb')
                restantes -= len(linha) + int(len(restante_paragrafo) > len(linha))
                restante_paragrafo = restante_paragrafo[len(linha):]
                y += 29
            y += 9
        for i, nome in enumerate(self.BOTOES):
            if i == self.PULAR and self.rever:
                nome = 'Sair da história'
            x = self.INICIO_BOTOES + i * self.LARGURA_BOTAO
            canto = (x * escala, 453 * escala, (x + 194) * escala, 491 * escala)
            desenho.rounded_rectangle(canto, radius=5 * escala, fill='#423341' if i == self.opcao else '#25202b',
                                      outline='#f3c880' if i == self.opcao else '#746172', width=escala)
            escrever(x + 12, 459, nome, 17, '#fff1d8' if i or self.pagina else '#aa9da8')
        for i in range(len(ABERTURA)):
            x = 50 + 15 * i
            desenho.ellipse((x * escala, 514 * escala, (x + 7) * escala, 521 * escala),
                            fill='#f3c880' if i == self.pagina else '#857588')
        escrever(160, 504, 'Indicador ou setas • Punho ou Enter • Espaço revela • Esc pula', 15, '#f3e5df')
        tela.pixels[:] = cv2.cvtColor(np.asarray(area), cv2.COLOR_RGB2BGR)
        fade = self.entrada if self.saida is None else self.saida
        if fade > 0:
            cv2.convertScaleAbs(tela.pixels, dst=tela.pixels, alpha=max(0, 1 - fade / self.DURACAO_FADE))


@lru_cache(maxsize=3)
def quadro_abertura(caminho, largura, altura):
    imagem = cv2.imdecode(np.fromfile(str(caminho), np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f'Cenário da introdução ilegível: {caminho}')
    quadro = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_LINEAR)
    sombra = np.linspace(.23, .43, largura, dtype=np.float32)[None, :, None]
    return np.rint(quadro * sombra).astype(np.uint8)
