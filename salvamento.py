"""Save local da expedição, com substituição atômica após cada marco."""
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


ARQUIVO = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Pedra Perdida' / 'progresso.json'


@dataclass
class Progresso:
    semente: int
    tutorial: int = 0
    fase: int = 0
    checkpoint: int = 80
    luzes: int = 0
    venceu: bool = False
    luzes_recolhidas: list[int] = field(default_factory=list)
    vidas_recolhidas: list[int] = field(default_factory=list)
    tutorial_pulado: bool = False

    def mapa_liberado(self, total_etapas):
        return self.tutorial_pulado or self.tutorial == total_etapas

    @classmethod
    def ler(cls, total_fases, total_etapas, arquivo=None):
        arquivo = ARQUIVO if arquivo is None else arquivo
        try:
            dados = json.loads(arquivo.read_text(encoding='utf-8'))
            if dados.pop('versao') != 1:
                return None
            save = cls(**dados)
            limites = ((save.semente, 0, 2**32 - 1), (save.tutorial, 0, total_etapas),
                       (save.fase, 0, total_fases - 1), (save.checkpoint, 80, 100_000),
                       (save.luzes, 0, 1_000_000))
            if any(type(v) is not int or not minimo <= v <= maximo for v, minimo, maximo in limites):
                return None
            for itens in (save.luzes_recolhidas, save.vidas_recolhidas):
                if not isinstance(itens, list) or any(type(i) is not int or i < 0 for i in itens):
                    return None
            if type(save.venceu) is bool and type(save.tutorial_pulado) is bool:
                return save
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def gravar(self, arquivo=None):
        arquivo = ARQUIVO if arquivo is None else arquivo
        temporario = None
        try:
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode='w', encoding='utf-8', dir=arquivo.parent, delete=False) as fluxo:
                temporario = Path(fluxo.name)
                json.dump(dict(versao=1, **asdict(self)), fluxo)
                fluxo.flush()
                os.fsync(fluxo.fileno())
            temporario.replace(arquivo)
        finally:
            if temporario is not None:
                temporario.unlink(missing_ok=True)
