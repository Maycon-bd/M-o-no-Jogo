"""Treinador pós-partida com Ollama em localhost e dica de contingência."""

import json
from concurrent.futures import Future
from threading import Thread
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

MODELO_SLM = "qwen2.5:0.5b"
ENDERECO = "http://127.0.0.1:11434/api/chat"


def orientacao(dados):
    if dados.get("vao_repetido"):
        return "Pegue impulso antes da borda e abra a mão para saltar.", ("impulso", "borda")
    if dados["segundos_sem_mao"] >= 3:
        return "Mantenha a mão inteira visível na câmera.", ("visível", "câmera", "quadro")
    if dados["trocas_de_direcao"] >= 4:
        return "Mantenha o mesmo gesto enquanto avança pelo caminho.", ("mant", "susten", "mesmo")
    if dados["segundos_teclado"] > 0:
        return "Use as setas para avançar pelo caminho.", ("seta", "tecla")
    return "Aponte com o indicador e mantenha o gesto enquanto anda.", ("indicador",)


def pedir_dica(dados):
    """Devolve a dica e identifica se veio do SLM ou da contingência."""
    instrucao, termos = orientacao(dados)
    esquema = {
        "type": "object",
        "properties": {"dica": {"type": "string"}},
        "required": ["dica"],
        "additionalProperties": False,
    }
    mensagem = {
        "model": MODELO_SLM,
        "stream": False,
        "format": esquema,
        "options": {"temperature": 0, "num_predict": 96},
        "messages": [
            {"role": "system", "content": (
                "Você é Mestre Granito, o sábio treinador do jogo Mão no Jogo. Adapte a orientação técnica "
                "à rodada em uma frase curta em português. Preserve o sentido "
                "da orientação e a ação que o jogador deve fazer. Não use números."
            )},
            {"role": "user", "content": (
                f"Orientação técnica: {instrucao}\n"
                f"Dados da rodada: {json.dumps(dados, ensure_ascii=False)}\n"
                "Escreva a dica para a próxima tentativa."
            )},
        ],
    }
    pedido = Request(ENDERECO, data=json.dumps(mensagem).encode("utf-8"),
                     headers={"Content-Type": "application/json"})
    try:
        with build_opener(ProxyHandler({})).open(pedido, timeout=18) as resposta:
            conteudo = json.load(resposta)["message"]["content"]
        texto = " ".join(json.loads(conteudo)["dica"].split())
        if (not 5 <= len(texto) <= 140 or any(c.isdigit() for c in texto)
                or not any(termo in texto.casefold() for termo in termos)):
            raise ValueError("Dica fora do formato esperado.")
        return texto, "SLM local"
    except (OSError, URLError, ValueError, KeyError, TypeError, AttributeError):
        return instrucao, "Dica de contingência"


def solicitar_dica(dados):
    resultado = Future()

    def gerar():
        resultado.set_result(pedir_dica(dados))

    Thread(target=gerar, daemon=True).start()
    return resultado


if __name__ == "__main__":
    exemplos = (
        {"pontos": 4, "segundos_sem_mao": 7, "trocas_de_direcao": 1, "segundos_teclado": 0},
        {"pontos": 4, "segundos_sem_mao": 0, "trocas_de_direcao": 8, "segundos_teclado": 0},
    )
    for exemplo in exemplos:
        dica, fonte = pedir_dica(exemplo)
        print(f"{fonte}: {dica}")
