"""Narração local do Mestre Granito, fora da thread de captura e renderização."""

import threading
import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory


MODELO_VOZ = Path(__file__).resolve().parent / 'modelos' / 'voz' / 'pt_BR-cadu-medium.onnx'


def sintetizar(voz, texto, destino):
    from piper import SynthesisConfig

    # Uma taxa de reprodução um pouco menor deixa a voz mais grave.
    configuracao = SynthesisConfig(length_scale=1.07, volume=.88)
    audio = voz.synthesize(texto, syn_config=configuracao)
    with wave.open(str(destino), 'wb') as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(round(voz.config.sample_rate * .92))
        for trecho in audio:
            arquivo.writeframes(trecho.audio_int16_bytes)
        return arquivo.getnframes() / arquivo.getframerate()


class VozGranito:
    def __init__(self, ativa=True):
        self.condicao = threading.Condition()
        self.pendente = None
        self.ultima = None
        self.ativa = False
        self.encerrada = False
        self.geracao = 0
        self.sintetizando = False
        self.termino = 0.0
        self.thread = None
        if ativa:
            self.ativar(True)

    def ativar(self, ativa):
        with self.condicao:
            if self.encerrada or ativa == self.ativa:
                return
            if ativa and not MODELO_VOZ.is_file():
                print('Voz do Mestre Granito indisponível. Veja a instalação no README.')
                return
            self.ativa = ativa
            if ativa and self.thread is None:
                self.thread = threading.Thread(target=self.narrar, daemon=True)
                self.thread.start()
            if not ativa:
                self._interromper()

    def _interromper(self):
        self.geracao += 1
        self.ultima = self.pendente = None
        self.termino = 0.0
        if self.thread is not None:
            from winsound import PlaySound
            PlaySound(None, 0)

    def interromper(self):
        with self.condicao:
            self._interromper()

    def falar(self, texto):
        with self.condicao:
            if self.ativa and not self.encerrada and texto != self.ultima:
                self.ultima = self.pendente = texto
                self.condicao.notify()

    def fechar(self):
        with self.condicao:
            self.encerrada = True
            self.condicao.notify()
        if self.thread is not None:
            self.thread.join(timeout=.35)

    def falando(self):
        with self.condicao:
            return self.ativa and not self.encerrada and (self.pendente is not None or self.sintetizando
                                           or time.monotonic() < self.termino)

    def narrar(self):
        try:
            import winsound
            from piper import PiperVoice

            voz = PiperVoice.load(MODELO_VOZ)
            with TemporaryDirectory(prefix='granito-') as pasta:
                arquivo = Path(pasta) / 'fala.wav'
                try:
                    while True:
                        with self.condicao:
                            self.condicao.wait_for(lambda: self.encerrada or self.pendente is not None)
                            if self.encerrada:
                                return
                            texto, self.pendente = self.pendente, None
                            geracao = self.geracao
                            self.sintetizando = True
                        winsound.PlaySound(None, 0)
                        duracao = sintetizar(voz, texto, arquivo)
                        with self.condicao:
                            self.sintetizando = False
                            if self.ativa and not self.encerrada and geracao == self.geracao and self.pendente is None:
                                winsound.PlaySound(str(arquivo), winsound.SND_FILENAME | winsound.SND_ASYNC)
                                self.termino = time.monotonic() + duracao
                finally:
                    winsound.PlaySound(None, 0)
        except Exception as erro:
            print(f'Voz do Mestre Granito indisponível: {erro}')
            with self.condicao:
                self.ativa = False
                self.encerrada = True
