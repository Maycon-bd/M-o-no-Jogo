"""Treina a CNN em PyTorch e exporta pesos usados pelo jogo sem PyTorch."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from cnn import CNN, PESOS_CNN, rasterizar
from gestos import GESTOS, MLP, MODELO
from treinar import carregar_csv, sintetizar


def avaliar(modelo, vetores, rotulos):
    matriz = np.zeros((len(GESTOS), len(GESTOS)), dtype=int)
    duracoes = []
    abstencoes = 0
    modelo.probabilidades(vetores[0])
    for vetor, gesto in zip(vetores, rotulos):
        inicio = perf_counter_ns()
        scores = modelo.probabilidades(vetor)
        duracoes.append((perf_counter_ns() - inicio) / 1_000_000)
        escolhido = modelo.classes[int(scores.argmax())]
        matriz[GESTOS.index(gesto), GESTOS.index(escolhido)] += 1
        abstencoes += int(scores.max() < 0.80)
    return {
        "acertos": int(matriz.trace()),
        "matriz": matriz.tolist(),
        "abaixo_de_080": abstencoes,
        "mediana_ms": round(float(np.median(duracoes)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--teste-pessoa", nargs="+")
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--mlp", type=Path, default=MODELO)
    parser.add_argument("--epocas", type=int, default=32)
    args = parser.parse_args()
    if bool(args.csv) != bool(args.teste_pessoa) or args.epocas < 1:
        parser.error("Use --csv junto de --teste-pessoa e escolha pelo menos uma época.")

    import torch
    from torch import nn

    torch.manual_seed(42)
    torch.set_num_threads(4)
    treino, y_treino = sintetizar(42, 400)
    teste, y_teste = sintetizar(117, 120, teste=True)
    if args.csv:
        vetores, rotulos, pessoas = carregar_csv(args.csv)
        if not set(args.teste_pessoa).issubset(pessoas):
            parser.error("Há um ID de teste ausente no CSV.")
        reservados = np.isin(pessoas, args.teste_pessoa)
        teste, y_teste = vetores[reservados], rotulos[reservados]
        if set(rotulos[~reservados]) != set(GESTOS) or set(y_teste) != set(GESTOS):
            parser.error("Os grupos de treino e teste precisam conter os três gestos.")
        treino = np.concatenate((treino, vetores[~reservados]))
        y_treino = np.concatenate((y_treino, rotulos[~reservados]))

    x = torch.from_numpy(np.stack([rasterizar(v) for v in treino])).float().unsqueeze(1) / 255
    y = torch.tensor([GESTOS.index(g) for g in y_treino])
    rede = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AvgPool2d(2), nn.Flatten(), nn.Linear(768, 3),
    )
    otimizador = torch.optim.Adam(rede.parameters(), lr=0.003)
    perda = nn.CrossEntropyLoss()
    for _ in range(args.epocas):
        for indices in torch.randperm(len(y)).split(64):
            otimizador.zero_grad()
            erro = perda(rede(x[indices]), y[indices])
            erro.backward()
            otimizador.step()

    padrao = PESOS_CNN.with_name("cnn_webcam.npz") if args.csv else PESOS_CNN
    saida = args.saida or padrao
    if saida.suffix.lower() != ".npz":
        parser.error("--saida precisa terminar em .npz.")
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("wb") as arquivo:
        np.savez_compressed(arquivo, **{
            chave: param.detach().numpy()
            for chave, param in zip(("w1", "b1", "w2", "b2", "w3", "b3"), rede.parameters())
        })

    cnn = CNN(saida)
    with torch.no_grad():
        amostra = torch.from_numpy(np.stack([rasterizar(v) for v in teste[:8]])).float()
        esperado = rede(amostra.unsqueeze(1) / 255).softmax(dim=1).numpy()
    np.testing.assert_allclose([cnn.probabilidades(v) for v in teste[:8]], esperado, atol=2e-6)
    resultado = {
        "origem_teste": "webcam, pessoas reservadas" if args.csv else "sintética",
        "amostras_treino_cnn": len(treino),
        "amostras_teste": len(teste),
        "gestos": GESTOS,
        "mlp_pesos": args.mlp.name,
        "cnn": avaliar(cnn, teste, y_teste),
        "mlp": avaliar(MLP(args.mlp), teste, y_teste),
    }
    relatorio = saida.with_name(f"avaliacao_{saida.stem}.json")
    relatorio.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
