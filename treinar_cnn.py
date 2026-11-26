"""Treina a CNN em PyTorch e exporta pesos usados pelo jogo sem PyTorch."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from cnn import CNN, PESOS_CNN, PESOS_CNN_APRIMORADA, rasterizar
from gestos import GESTOS, MLP, MODELO, normalizar
from treinar import carregar_csv, sintetizar


def avaliar(modelo, vetores, rotulos):
    matriz = np.zeros((len(GESTOS), len(GESTOS)), dtype=int)
    duracoes = []
    abstencoes = 0
    aceitos_corretos = 0
    modelo.probabilidades(vetores[0])
    for vetor, gesto in zip(vetores, rotulos):
        inicio = perf_counter_ns()
        scores = modelo.probabilidades(vetor)
        duracoes.append((perf_counter_ns() - inicio) / 1_000_000)
        escolhido = modelo.classes[int(scores.argmax())]
        matriz[GESTOS.index(gesto), GESTOS.index(escolhido)] += 1
        if scores.max() < 0.80:
            abstencoes += 1
        elif escolhido == gesto:
            aceitos_corretos += 1
    acertos = int(matriz.trace())
    return {
        "acertos": acertos,
        "acuracia": acertos / len(rotulos),
        "matriz": matriz.tolist(),
        "abaixo_de_080": abstencoes,
        "aceitos_corretos": aceitos_corretos,
        "mediana_ms": round(float(np.median(duracoes)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--teste-pessoa", nargs="+")
    parser.add_argument("--validacao-pessoa", nargs="+")
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--mlp", type=Path, default=MODELO)
    parser.add_argument("--epocas", type=int, default=32)
    parser.add_argument("--amostras-por-gesto", type=int, default=1500)
    parser.add_argument("--taxa", type=float, default=0.003)
    parser.add_argument("--referencia", type=Path, default=PESOS_CNN)
    parser.add_argument("--faixa-validacao", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="seleciona a época mais próxima do centro da faixa de acurácia")
    args = parser.parse_args()
    if not (bool(args.csv) == bool(args.teste_pessoa) == bool(args.validacao_pessoa)):
        parser.error("Use --csv junto de --validacao-pessoa e --teste-pessoa.")
    if args.epocas < 1 or args.amostras_por_gesto < 1 or not np.isfinite(args.taxa) or args.taxa <= 0:
        parser.error("Épocas, amostras por gesto e taxa devem ser positivas.")
    faixa = args.faixa_validacao
    if faixa and (not 0 <= faixa[0] <= faixa[1] <= 1 or args.saida is None):
        parser.error("Com --faixa-validacao, use limites entre 0 e 1 e informe --saida.")
    padrao = PESOS_CNN.with_name("cnn_webcam.npz") if args.csv else PESOS_CNN_APRIMORADA
    saida = args.saida or padrao
    if saida.suffix.lower() != ".npz" or saida.resolve() == args.referencia.resolve():
        parser.error("Use uma saída .npz diferente do modelo de referência.")

    treino, y_treino = sintetizar(42, args.amostras_por_gesto, ampliar=True)
    validacao, y_validacao = sintetizar(83, 200, teste=True)
    teste, y_teste = sintetizar(117, 120, teste=True)
    if args.csv:
        vetores, rotulos, pessoas = carregar_csv(args.csv)
        ids_val, ids_teste = set(args.validacao_pessoa), set(args.teste_pessoa)
        if ids_val & ids_teste or not (ids_val | ids_teste).issubset(pessoas):
            parser.error("Validação e teste devem ter IDs existentes e diferentes.")
        mascara_val = np.isin(pessoas, args.validacao_pessoa)
        mascara_teste = np.isin(pessoas, args.teste_pessoa)
        mascara_treino = ~(mascara_val | mascara_teste)
        for mascara in (mascara_treino, mascara_val, mascara_teste):
            if set(rotulos[mascara]) != set(GESTOS):
                parser.error("Os grupos de treino, validação e teste precisam conter os três gestos.")
        validacao, y_validacao = vetores[mascara_val], rotulos[mascara_val]
        teste, y_teste = vetores[mascara_teste], rotulos[mascara_teste]
        reais, y_reais = vetores[mascara_treino], rotulos[mascara_treino]
        espelhados = reais.reshape(-1, 21, 3).copy()
        espelhados[:, :, 0] *= -1
        invertidos = [{"direita": "esquerda", "esquerda": "direita"}.get(g, g) for g in y_reais]
        rng = np.random.default_rng(91)
        variados = espelhados + rng.normal(0, .015, espelhados.shape)
        treino = np.concatenate((treino, reais, espelhados.reshape(-1, 63),
                                 np.stack([normalizar(p) for p in variados])))
        y_treino = np.concatenate((y_treino, y_reais, invertidos, invertidos))

    import torch
    from torch import nn

    torch.manual_seed(42)
    torch.set_num_threads(4)
    x = torch.from_numpy(np.stack([rasterizar(v) for v in treino])).float().unsqueeze(1) / 255
    y = torch.tensor([GESTOS.index(g) for g in y_treino])
    x_val = torch.from_numpy(np.stack([rasterizar(v) for v in validacao])).float().unsqueeze(1) / 255
    y_val = torch.tensor([GESTOS.index(g) for g in y_validacao])
    rede = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AvgPool2d(2), nn.Flatten(), nn.Linear(768, 3),
    )
    otimizador = torch.optim.Adam(rede.parameters(), lr=args.taxa)
    perda = nn.CrossEntropyLoss()
    agenda = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, factor=.5, patience=3)
    melhor_perda = float("inf")
    melhor_epoca = 0
    melhor_acuracia = None
    melhor_criterio = (float("inf"), float("inf"))
    centro = sum(faixa) / 2 if faixa else None
    for epoca in range(1, args.epocas + 1):
        for indices in torch.randperm(len(y)).split(64):
            otimizador.zero_grad()
            erro = perda(rede(x[indices]), y[indices])
            erro.backward()
            otimizador.step()
        with torch.no_grad():
            previsoes = rede(x_val)
            perda_val = perda(previsoes, y_val).item()
            acuracia_val = int((previsoes.argmax(dim=1) == y_val).sum().item()) / len(y_val)
        agenda.step(perda_val)
        if faixa:
            criterio = (abs(acuracia_val - centro), perda_val)
            selecionada = faixa[0] <= acuracia_val <= faixa[1] and criterio < melhor_criterio
        else:
            selecionada = perda_val < melhor_perda - 1e-4
        if selecionada:
            melhor_perda = perda_val
            if faixa:
                melhor_criterio = criterio
            melhor_acuracia = acuracia_val
            melhor_epoca = epoca
            pesos = {chave: tensor.clone() for chave, tensor in rede.state_dict().items()}
        if faixa or epoca % 4 == 0 or epoca == args.epocas:
            print(f"Época {epoca}/{args.epocas} | validação {acuracia_val:.2%} | perda {perda_val:.4f}", flush=True)
        if not faixa and epoca - melhor_epoca >= 9:
            break

    if melhor_epoca == 0:
        raise SystemExit("Nenhuma época atingiu o critério de validação. Nenhum peso foi gravado.")
    rede.load_state_dict(pesos)
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
        "amostras_validacao": len(validacao),
        "amostras_teste": len(teste),
        "melhor_epoca": melhor_epoca,
        "acuracia_validacao": melhor_acuracia,
        "faixa_validacao": faixa,
        "epocas_executadas": epoca,
        "taxa_inicial": args.taxa,
        "pessoas_validacao": args.validacao_pessoa,
        "pessoas_teste": args.teste_pessoa,
        "pesos_referencia": args.referencia.name,
        "gestos": GESTOS,
        "mlp_pesos": args.mlp.name,
        "cnn": avaliar(cnn, teste, y_teste),
        "cnn_referencia": avaliar(CNN(args.referencia), teste, y_teste),
        "mlp": avaliar(MLP(args.mlp), teste, y_teste),
    }
    if not args.csv:
        extra, y_extra = sintetizar(901, 300, teste=True)
        resultado["teste_sintetico_independente"] = {
            "semente": 901,
            "amostras": len(extra),
            "cnn": avaliar(cnn, extra, y_extra),
            "cnn_referencia": avaliar(CNN(args.referencia), extra, y_extra),
            "mlp": avaliar(MLP(args.mlp), extra, y_extra),
        }
    relatorio = saida.with_name(f"avaliacao_{saida.stem}.json")
    relatorio.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
