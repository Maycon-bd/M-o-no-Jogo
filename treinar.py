"""Coleta coordenadas, treina a MLP e mede o desempenho em um conjunto separado."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from gestos import GESTOS, MODELO, RAIZ, MLP, Webcam, desenhar, normalizar

RELATORIO = RAIZ / "modelos" / "avaliacao_sintetica.json"
COLUNAS = ["pessoa", "gesto"] + [f"p{i}_{eixo}" for i in range(21) for eixo in "xyz"]

# Índices iguais aos do Hand Landmarker. A orientação inicial aponta para cima.
PALMA = np.array([
    (0, 0, 0), (-.27, -.16, -.02), (-.46, -.30, -.04),
    (-.60, -.42, -.06), (-.72, -.53, -.08),
    (-.26, -.54, -.02), (-.31, -.82, -.03), (-.34, -1.00, -.04), (-.36, -1.16, -.04),
    (-.06, -.59, 0), (-.08, -.91, 0), (-.09, -1.12, 0), (-.10, -1.29, 0),
    (.14, -.56, .02), (.18, -.84, .02), (.21, -1.03, .03), (.24, -1.18, .03),
    (.31, -.48, .04), (.39, -.69, .05), (.44, -.84, .05), (.47, -.96, .05),
], dtype=np.float32)


def sintetizar(semente, por_gesto, teste=False):
    rng = np.random.default_rng(semente)
    entradas, rotulos = [], []
    for gesto in GESTOS:
        for _ in range(por_gesto):
            pontos = PALMA.copy()
            if gesto != "parar":
                for inicio in (9, 13, 17):
                    curvatura = rng.uniform(.65, 1.12) if teste else rng.uniform(.72, 1.02)
                    pontos[inicio + 1:inicio + 4] = pontos[inicio] + np.array([
                        (.02, -.21, -.07), (.03, -.23, -.23), (.00, -.06, -.30),
                    ]) * curvatura
                pontos[2:5] += [(0, -.04, -.04), (.20, -.08, -.12), (.45, -.10, -.20)]
            pontos[:, 0] *= rng.choice((-1, 1)) * rng.uniform(.65, 1.24)
            pontos[:, 1] *= rng.uniform(.86, 1.13)
            if gesto == "parar":
                angulo = rng.uniform(-180, 180)
            else:
                direcao = -1 if gesto == "esquerda" else 1
                abertura = 29 if teste else 23
                angulo = direcao * 90 + rng.uniform(-abertura, abertura)
            seno, cosseno = np.sin(np.deg2rad(angulo)), np.cos(np.deg2rad(angulo))
            pontos[:, :2] = pontos[:, :2] @ [[cosseno, seno], [-seno, cosseno]]
            pontos += rng.normal(0, .042 if teste else .026, pontos.shape)
            entradas.append(normalizar(pontos))
            rotulos.append(gesto)
    return np.asarray(entradas), np.asarray(rotulos)


def coletar(args):
    import cv2

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    arquivo_novo = not args.csv.exists() or args.csv.stat().st_size == 0
    janela = "Coleta | 1 esquerda, 2 direita, 3 parar | Espaco registra | Esc sai"
    contagens = dict.fromkeys(GESTOS, 0)
    gesto = "parar"
    try:
        with Webcam(args.camera) as webcam, args.csv.open(
            "a", encoding="utf-8", newline=""
        ) as arquivo:
            escritor = csv.writer(arquivo)
            if arquivo_novo:
                escritor.writerow(COLUNAS)
            while True:
                frame, vetor, pontos = webcam.ler()
                if frame is None:
                    raise OSError("A webcam deixou de enviar quadros.")
                if pontos is not None:
                    desenhar(cv2, frame, pontos)
                texto = f"{gesto.upper()} | amostras da sessao: {contagens[gesto]}"
                cv2.putText(frame, texto, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(janela, frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == 27 or cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if tecla in (ord("1"), ord("2"), ord("3")):
                    gesto = GESTOS[tecla - ord("1")]
                if tecla == ord(" ") and vetor is not None:
                    escritor.writerow([args.pessoa, gesto, *vetor])
                    arquivo.flush()
                    contagens[gesto] += 1
    finally:
        cv2.destroyAllWindows()
    print(contagens)


def carregar_csv(arquivo):
    with arquivo.open(encoding="utf-8", newline="") as entrada:
        linhas = list(csv.DictReader(entrada))
    if not linhas:
        raise ValueError("O CSV precisa conter amostras coletadas.")
    grupos = np.array([linha["pessoa"] for linha in linhas])
    labels = np.array([linha["gesto"] for linha in linhas])
    if not set(labels).issubset(GESTOS):
        raise ValueError("O CSV tem rótulos que não fazem parte dos três gestos.")
    dados = np.array([[float(linha[k]) for k in COLUNAS[2:]] for linha in linhas])
    if not np.isfinite(dados).all():
        raise ValueError("O CSV tem coordenadas inválidas.")
    return dados, labels, grupos


def treinar_avaliar(args):
    from sklearn.metrics import confusion_matrix
    from sklearn.neural_network import MLPClassifier

    if args.csv:
        dados, rotulos, pessoas = carregar_csv(args.csv)
        reservados = np.isin(pessoas, args.teste_pessoa)
        if not set(args.teste_pessoa).issubset(pessoas):
            raise ValueError("Pelo menos um ID de teste não está no CSV.")
        treino, y_treino = dados[~reservados], rotulos[~reservados]
        teste, y_teste = dados[reservados], rotulos[reservados]
    else:
        treino, y_treino = sintetizar(42, 400)
        teste, y_teste = sintetizar(117, 120, teste=True)
    if set(y_treino) != set(GESTOS) or set(y_teste) != set(GESTOS):
        raise ValueError("Treino e teste precisam conter exemplos dos três gestos.")
    rede = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                         max_iter=250, random_state=42)
    rede.fit(treino, y_treino)
    args.modelo.parent.mkdir(parents=True, exist_ok=True)
    artefato = {
        "classes": rede.classes_.tolist(),
        "pesos": [camada.tolist() for camada in rede.coefs_],
        "vieses": [camada.tolist() for camada in rede.intercepts_],
    }
    args.modelo.write_text(json.dumps(artefato), encoding="utf-8")

    inferencia = MLP(args.modelo)
    probabilidades = inferencia.probabilidades(teste)
    np.testing.assert_allclose(probabilidades, rede.predict_proba(teste), atol=2e-6)
    classes = np.array(inferencia.classes)
    predicoes = classes[probabilidades.argmax(axis=1)]
    matriz = confusion_matrix(y_teste, predicoes, labels=GESTOS)
    aceitos = probabilidades.max(axis=1) >= .80
    passos = []
    for vetor in teste:
        inicio = time.perf_counter_ns()
        inferencia.probabilidades(vetor)
        passos.append((time.perf_counter_ns() - inicio) / 1_000_000)
    resultado = {
        "origem": "webcam, separados por participante" if args.csv else "sintética",
        "amostras_treino": len(treino),
        "amostras_teste": len(teste),
        "gestos": GESTOS,
        "matriz": matriz.tolist(),
        "acertos": int(np.trace(matriz)),
        "acuracia": float(np.mean(predicoes == y_teste)),
        "abaixo_de_080": int(np.sum(~aceitos)),
        "participantes_teste": len(args.teste_pessoa or []),
        "mediana_mlp_ms": round(float(np.median(passos)), 4),
    }
    args.relatorio.parent.mkdir(parents=True, exist_ok=True)
    args.relatorio.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="acao", required=True)
    coleta = sub.add_parser("coletar")
    coleta.add_argument("--csv", type=Path, required=True)
    coleta.add_argument("--pessoa", required=True)
    coleta.add_argument("--camera", type=int, default=0)
    treino = sub.add_parser("treinar")
    treino.add_argument("--csv", type=Path)
    treino.add_argument("--teste-pessoa", nargs="+")
    treino.add_argument("--modelo", type=Path)
    treino.add_argument("--relatorio", type=Path)
    args = parser.parse_args()
    if args.acao == "coletar":
        coletar(args)
        return
    if bool(args.csv) != bool(args.teste_pessoa):
        parser.error("Use --csv e --teste-pessoa juntos para avaliar dados de webcam.")
    args.modelo = args.modelo or (MODELO.with_name("mlp_webcam.json") if args.csv else MODELO)
    args.relatorio = args.relatorio or (
        RELATORIO.with_name("avaliacao_webcam.json") if args.csv else RELATORIO
    )
    treinar_avaliar(args)


if __name__ == "__main__":
    main()
