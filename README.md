# Mão no Jogo

- 5º Workshop de Trabalhos de IA da FASOFT · UniRV
- Redes Neurais · Professor Sandro Silva Moreira
- Maycon Garcia Silva e Danilo Da Silva Faria

## Como rodar no Windows

Abra o PowerShell nesta pasta com **Python 3.12** disponível. Instale e abra o
[Ollama para Windows](https://ollama.com/download/windows), depois execute:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:0.5b
.\.venv\Scripts\python.exe jogo.py
```

Na janela, aponte o indicador na direção desejada com os outros dedos
recolhidos. Mostre a palma aberta para parar. A rodada dura **20 segundos**.
No fim, o treinador mostra a dica e a origem dela. **Espaço/R** reinicia e
**Esc** sai. Depois da instalação e do download inicial, mantenha o Ollama
aberto e rode o jogo com a internet desligada.

| Comando após `jogo.py` | Uso |
| --- | --- |
| `--camera 1` | Tenta outra câmera. |
| `--teclado` | Joga com as setas. Se a câmera falhar na abertura ou durante a rodada, o jogo também muda para esse modo. |
| `--mlp` | Troca a CNN pela MLP original para a demonstração comparativa. |
| `--modelo modelos/cnn_webcam.npz` | Usa pesos da CNN reentreinados com amostras reais. Arquivos JSON continuam sendo pesos da MLP. |
| `--sem-treinador` | Joga com a tela final de métricas e sem consultar o Ollama. |

Para verificar só o Qwen, execute `.\.venv\Scripts\python.exe treinador.py`.
Devem aparecer duas linhas identificadas como **SLM local**. Se aparecer
**Dica de contingência**, confira se o Ollama está aberto e se
`ollama list` mostra `qwen2.5:0.5b`.

## Como a aplicação funciona

1. O OpenCV espelha a imagem e o MediaPipe encontra 21 pontos da mão.
2. O jogo transforma os pontos em uma imagem monocromática de 64 × 64 pixels
   e mostra essa entrada ao lado da câmera. A CNN usa duas convoluções 3 × 3,
   ReLU, pooling e uma camada de três saídas. O filtro confirma um gesto com
   probabilidade de pelo menos 0,80 em três quadros consecutivos.
3. Ao fim da rodada, o jogo mede placar, tempo sem mão visível, tempo usando
   teclado e trocas de direção reconhecidas. O Qwen recebe esses quatro
   valores e uma indicação do aspecto a comentar via API local do Ollama.
   A classificação e o jogo seguem disponíveis com uma dica de contingência
   quando a consulta falha.

Os pesos iniciais foram treinados com dados sintéticos. O relatório
`modelos/avaliacao_cnn.json` compara CNN e MLP nos mesmos 360 vetores de
teste sintético. O tempo da CNN inclui a rasterização. Os tempos excluem a
captura e o MediaPipe. O Qwen é pré-treinado. A aplicação usa instruções e uma
saída JSON curta para gerar a dica.

## Testes e coleta com webcam

O teste automatizado roda sem câmera. O diagnóstico por webcam usa a MLP de
referência e desenha os pontos da mão. Feche a janela antes de iniciar o jogo.

```powershell
.\.venv\Scripts\python.exe teste_jogo.py
.\.venv\Scripts\python.exe gestos.py
```

Para medir generalização, colete pelo menos 20 poses independentes por gesto
e por pessoa, com as três classes em cada grupo. As teclas **1/2/3** escolhem
o gesto, **Espaço** registra os pontos e **Esc** encerra. Reserve a última
pessoa para teste e use outras pessoas no treino. PyTorch só é necessário
para reentreinar:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe treinar.py coletar --csv dados/gestos.csv --pessoa P01
.\.venv\Scripts\python.exe treinar.py coletar --csv dados/gestos.csv --pessoa P02
.\.venv\Scripts\python.exe treinar_cnn.py --csv dados/gestos.csv --teste-pessoa P02
.\.venv\Scripts\python.exe jogo.py --modelo modelos/cnn_webcam.npz
```

Nesse ensaio, a CNN é treinada com os vetores sintéticos e as pessoas de
treino, e a MLP original é a referência. A mesma pessoa reservada é usada
para medir os dois modelos. A avaliação é gravada em
`modelos/avaliacao_cnn_webcam.json`. Para reproduzir o ensaio sintético,
rode `treinar_cnn.py` sem argumentos.

## Materiais

- `docs/Relatorio tecnico - Maycon.docx` e `docs/Guia de integracao - Danilo.docx` documentam a etapa anterior com MLP. A arquitetura atual está descrita neste README.
- `banner/Banner - Mao no Jogo.pdf` é o banner em 90 × 120 cm.
- `banner/Banner editavel - Mao no Jogo.svg` permite atualizar texto e figuras em um editor vetorial. Exporte em 90 × 120 cm após editar.

Para fechar a versão de exposição, teste a webcam e o SLM offline no notebook
da bancada e atualize o banner com os dados de participantes que conseguir medir.

## Título e resumo para o cadastro

**Mão no Jogo — Controle de um minijogo por gestos com redes neurais**

O Mão no Jogo é um minijogo interativo controlado por gestos capturados pela
webcam. Uma rede neural treinada para o projeto classifica três gestos da mão e
os converte em comandos para mover ou parar o personagem. Durante a exposição,
os visitantes poderão jogar em tempo real e acompanhar na tela qual gesto foi
reconhecido. A aplicação será executada localmente para funcionar sem internet.
