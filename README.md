# Mão no Jogo

- 5º Workshop de Trabalhos de IA da FASOFT · UniRV
- Redes Neurais · Professor Sandro Silva Moreira
- Maycon Garcia Silva e Danilo Da Silva Faria

## Diagnóstico do Maycon

No PowerShell, abra esta pasta e execute com Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe gestos.py
```

Use `--camera 1` se a câmera estiver em outro índice. A janela mostra a mão
espelhada, o comando e o estado. Aponte o indicador na direção desejada
com os demais dedos recolhidos ou mostre a palma aberta para parar.
Pressione Esc para sair. Feche a janela de diagnóstico antes de abrir o jogo.

O modelo do Google e os pesos iniciais estão em `modelos/`. Depois da
instalação, a captura, a detecção e a classificação ocorrem localmente.
O treinamento inicial utiliza dados sintéticos. A avaliação correspondente
fica em `modelos/avaliacao_sintetica.json`.

## Materiais

- `docs/Relatorio tecnico - Maycon.docx` explica arquitetura, treinamento e validação.
- `docs/Guia de integracao - Danilo.docx` descreve a API, o trabalho do Danilo e o ensaio.
- `banner/Banner - Mao no Jogo.pdf` é o banner em 90 × 120 cm.
- `banner/Banner editavel - Mao no Jogo.svg` permite atualizar texto e figuras em um editor vetorial. Exporte em 90 × 120 cm após editar.

O banner registra a avaliação sintética e traz uma ilustração do jogo.
Para fechar a versão de exposição, a dupla pode incluir o teste com participantes,
uma captura do minijogo integrado e os e-mails acadêmicos no cabeçalho.

## Título e resumo para o cadastro

**Mão no Jogo — Controle de um minijogo por gestos com redes neurais**

O Mão no Jogo é um minijogo interativo controlado por gestos capturados pela
webcam. Uma rede neural treinada para o projeto classifica três gestos da mão e
os converte em comandos para mover ou parar o personagem. Durante a exposição,
os visitantes poderão jogar em tempo real e acompanhar na tela qual gesto foi
reconhecido. A aplicação será executada localmente para funcionar sem internet.
