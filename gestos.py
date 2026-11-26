"""Controlador local de gestos e visualizador de diagnóstico."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
LANDMARKER = RAIZ / "modelos" / "hand_landmarker.task"
MODELO = RAIZ / "modelos" / "mlp.json"
GESTOS = ("esquerda", "direita", "parar")
DEDOS = [
    (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17),
]


def normalizar(pontos):
    """Centraliza no pulso e divide pelo maior raio no plano da imagem."""
    relativos = np.asarray(pontos, dtype=np.float32) - pontos[0]
    raio = np.linalg.norm(relativos[:, :2], axis=1).max()
    return (relativos / max(raio, 1e-6)).ravel()


@dataclass(frozen=True)
class Leitura:
    comando: str
    estado: str
    confianca: float = 0.0


class MLP:
    def __init__(self, arquivo=MODELO):
        dados = json.loads(Path(arquivo).read_text(encoding="utf-8"))
        self.camadas = [
            (np.asarray(peso, np.float32), np.asarray(vies, np.float32))
            for peso, vies in zip(dados["pesos"], dados["vieses"])
        ]
        self.classes = dados["classes"]

    def probabilidades(self, entrada):
        for peso, vies in self.camadas[:-1]:
            entrada = np.maximum(entrada @ peso + vies, 0)
        peso, vies = self.camadas[-1]
        logits = entrada @ peso + vies
        expoentes = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return expoentes / expoentes.sum(axis=-1, keepdims=True)


class Estabilizador:
    def __init__(self, limiar=0.80, quadros=3):
        self.limiar = limiar
        self.quadros = quadros
        self.reiniciar()

    def reiniciar(self):
        self.candidato = None
        self.contagem = 0

    def atualizar(self, gesto, confianca):
        if confianca < self.limiar:
            self.reiniciar()
            return Leitura("parar", "incerto", confianca)
        if gesto != self.candidato:
            self.candidato = gesto
            self.contagem = 0
        self.contagem += 1
        if self.contagem < self.quadros:
            return Leitura("parar", "estabilizando", confianca)
        return Leitura(gesto, "reconhecido", confianca)


class Webcam:
    """Um consumidor da câmera. Todos os frames retornados já estão espelhados."""

    def __init__(self, indice=0):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not LANDMARKER.is_file():
            raise OSError(f"Ativo MediaPipe ausente: {LANDMARKER}")
        self.cv2 = cv2
        self.mp = mp
        self.camera = cv2.VideoCapture(indice)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.camera.isOpened():
            self.camera.release()
            raise OSError(f"Não foi possível abrir a webcam {indice}.")
        try:
            self.detector = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(LANDMARKER)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=1,
                )
            )
        except Exception:
            self.camera.release()
            raise
        self.timestamp = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.camera.release()
        self.detector.close()

    def ler(self):
        ok, frame = self.camera.read()
        if not ok:
            return None, None, None
        frame = self.cv2.flip(frame, 1)
        imagem = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB),
        )
        self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
        resultado = self.detector.detect_for_video(imagem, self.timestamp)
        if not resultado.hand_landmarks:
            return frame, None, None
        altura, largura = frame.shape[:2]
        pontos = np.array(
            [(p.x * largura, p.y * altura, p.z * largura)
             for p in resultado.hand_landmarks[0]],
            dtype=np.float32,
        )
        return frame, normalizar(pontos), pontos


class ControleGestos:
    """API do jogo: com o contexto aberto, leia um comando por iteração."""

    def __init__(self, indice=0, modelo=MODELO, classificador=None):
        self.classificador = classificador if classificador is not None else MLP(modelo)
        self.filtro = Estabilizador()
        self.webcam = Webcam(indice)
        self.pontos = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.webcam.__exit__(*args)

    def ler(self):
        frame, entrada, pontos = self.webcam.ler()
        self.pontos = pontos
        if entrada is None:
            if hasattr(self.classificador, "imagem"):
                self.classificador.imagem = None
            self.filtro.reiniciar()
            estado = "sem_mao" if frame is not None else "falha_camera"
            return Leitura("parar", estado), frame
        probabilidades = self.classificador.probabilidades(entrada)
        indice = int(probabilidades.argmax())
        leitura = self.filtro.atualizar(
            self.classificador.classes[indice], float(probabilidades[indice])
        )
        desenhar(self.webcam.cv2, frame, pontos)
        return leitura, frame


def desenhar(cv2, frame, pontos):
    xy = np.rint(pontos[:, :2]).astype(int)
    for caminho in DEDOS:
        for inicio, fim in zip(caminho, caminho[1:]):
            cv2.line(frame, tuple(xy[inicio]), tuple(xy[fim]), (255, 210, 55), 2)
    for ponto in xy:
        cv2.circle(frame, tuple(ponto), 4, (60, 230, 255), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    import cv2

    try:
        with ControleGestos(args.camera) as controle:
            while True:
                leitura, frame = controle.ler()
                if frame is None:
                    raise OSError("A webcam deixou de enviar quadros.")
                texto = f"{leitura.comando} | {leitura.estado} | {leitura.confianca:.0%}"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (25, 25, 25), -1)
                cv2.putText(frame, texto, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Mao no Jogo - diagnostico | Esc sai", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if cv2.getWindowProperty("Mao no Jogo - diagnostico | Esc sai",
                                         cv2.WND_PROP_VISIBLE) < 1:
                    break
    except OSError as erro:
        raise SystemExit(str(erro)) from erro
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
