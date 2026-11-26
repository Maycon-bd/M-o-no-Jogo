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
.\.venv\Scripts\python.exe -m piper.download_voices pt_BR-cadu-medium --data-dir modelos/voz
ollama pull qwen2.5:0.5b
.\.venv\Scripts\python.exe jogo.py
```

Escolha **Play** para abrir o modo História. Na primeira vez, a abertura
leva à Fase 00, O Berço das Pedras. Se já houver um save, o modo História
oferece **Continuar**, **Novo Jogo** e **Rever história**. **Tutorial** abre
o Berço para praticar sem alterar a campanha. **Créditos** exibe a lista
rolante de direção, referências e design.

Aponte para cima ou para baixo no menu inicial e feche a mão para confirmar.
Setas, Enter e mouse também funcionam. Na abertura, Espaço revela o texto.
**Pular e ir ao mapa** ou Esc libera a primeira caverna e abre o mapa direto.
**Rever história** reproduz a abertura e volta ao título.

No Berço, o personagem Preda, uma pedra animada, encontra Mestre Granito.
Avance as falas com o punho,
Enter ou clique. Para progredir, role até a marca, pouse na plataforma,
acelere na descida e confirme o Portão das Profundezas. A barra registra
seis etapas. No fim da missão, a Fase 01 é liberada no mapa.

**Continuar** carrega o último marco do tutorial ou a fase liberada e seu
checkpoint. A semente, o progresso e as luzes dos checkpoints são gravados
em `%LOCALAPPDATA%\Pedra Perdida\progresso.json`. Após **Game Over**,
o título volta automaticamente. Ao continuar, você retorna com três vidas.
O asset da Fase 00 é `assets/berco-das-pedras.png`. Se existir
`assets/00-O-Berco-das-Pedras-Tutorial.png`, ele tem prioridade.

O Mestre Granito narra suas falas com a voz brasileira Cadu do Piper,
ajustada para um tom mais grave e pausado. Depois de baixar o modelo, a
síntese funciona localmente, inclusive nas dicas do SLM.
[Modelo Cadu](https://huggingface.co/rhasspy/piper-voices/blob/main/pt/pt_BR/cadu/medium/MODEL_CARD)

Durante a partida, aponte o indicador para os lados para
rolar. Incline o indicador para controlar a velocidade. Quanto mais perto da
horizontal, maior a velocidade pedida. Apontar para cima mantém o embalo sem
acelerar. Abra a mão para pular, mantendo o impulso. Feche a mão para frear
gradualmente. Para abrir o menu de pausa, mantenha indicador, médio e anelar
estendidos, com o mindinho recolhido. Aponte para os lados para percorrer as
opções e feche a mão para confirmar.
O menu de pausa permite continuar, voltar ao mapa ou ao título. No tutorial,
a pausa também permite voltar ao marco da etapa atual.
O mapa ilustrado dá acesso às oito fases procedurais. Aponte na direção do
próximo disco ou use a seta direita para avançar e a esquerda para voltar.
Preda rola pela trilha entre as
entradas liberadas. Feche a mão, pressione Enter ou clique em
**Entrar** depois que ele parar. Chegar ao portal libera a próxima entrada.
É possível revisitar fases concluídas. Nessas tentativas, as luzes, vidas e
progresso da campanha ficam guardados até o retorno ao mapa.

O caminho passa pelo Portão das Profundezas, Lago das Lanternas, Vale das
Ametistas, Rio de Magma, Coração dos Geodos, Jardim Bioluminescente, Ossário
de Pedra e Abismo de Obsidiana. O fundo ilustrado de cada fase acompanha a
câmera e se repete com um degradê entre as cópias espelhadas. O chão e as
plataformas jogáveis seguem o percurso procedural.

O gerador combina de oito a quinze trechos por fase, com vãos progressivos,
patrulhas e dois pontos de retorno. Encoste nos inimigos em alta velocidade
ou caia por cima para derrotá-los. Um encontrão devagar custa uma vida.
A barra de velocidade indica quando o Rocha está pronto para atacar.

Para pegar uma vida extra, salte até a rampa suspensa, role em direção ao gêiser
e atravesse o jato quando o indicador mostrar **JÁ!**. Mantenha o impulso para
chegar ao patamar com o coração e as luzes. É possível passar por baixo da rampa
e seguir pela rota principal.

O gerador simula o percurso obrigatório com três pontos de salto antes de
aceitar cada fase. A opção `--semente 18947` abre um percurso reproduzível
para teste e mantém o save da campanha separado.
O salto aceita a palma até 120 ms depois de sair da borda ou até 150 ms antes
de pousar. Duas quedas no mesmo vão acionam uma dica contextual do Mestre Granito.
Atravesse as oito fases,
colete luzes e alcance os portais, evitando buracos e espinhos. Você tem três
vidas e encontra pontos de retorno durante as fases.
Também é possível usar setas para andar, Espaço para pular, F para frear e M
ou Esc para abrir a pausa. No título, Esc fecha o jogo.
Depois da instalação e do download inicial, mantenha o Ollama aberto e rode o
jogo com a internet desligada.

| Comando após `jogo.py` | Uso |
| --- | --- |
| `--camera 1` | Tenta outra câmera. |
| `--teclado` | Joga com as setas. Se a câmera falhar na abertura ou durante a rodada, o jogo também muda para esse modo. |
| `--mlp` | Troca a CNN pela MLP original para a demonstração comparativa. |
| `--modelo modelos/cnn_webcam.npz` | Usa pesos da CNN reentreinados com amostras reais. Arquivos JSON continuam sendo pesos da MLP. |
| `--sem-treinador` | Joga com a tela final de métricas e sem consultar o Ollama. |
| `--sem-voz` | Silencia a narração do Mestre Granito. |

Para verificar só o Qwen, execute `.\.venv\Scripts\python.exe treinador.py`.
Devem aparecer duas linhas identificadas como **SLM local**. Se aparecer
**Dica de contingência**, confira se o Ollama está aberto e se
`ollama list` mostra `qwen2.5:0.5b`.

## Como a aplicação funciona

1. O OpenCV espelha a imagem e o MediaPipe encontra 21 pontos da mão.
   São o pulso, três articulações e a ponta de cada um dos cinco dedos.
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
o gesto, **Espaço** registra os pontos e **Esc** encerra. Separe pessoas para
treino, validação e teste. PyTorch só é necessário
para reentreinar:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe treinar.py coletar --csv dados/gestos.csv --pessoa P01
.\.venv\Scripts\python.exe treinar.py coletar --csv dados/gestos.csv --pessoa P02
.\.venv\Scripts\python.exe treinar.py coletar --csv dados/gestos.csv --pessoa P03
.\.venv\Scripts\python.exe treinar_cnn.py --csv dados/gestos.csv --validacao-pessoa P02 --teste-pessoa P03 --amostras-por-gesto 400
.\.venv\Scripts\python.exe jogo.py --modelo modelos/cnn_webcam.npz
```

A CNN usa dados sintéticos e amostras reais de treino, incluindo versões
espelhadas com a direção corrigida. A validação escolhe os pesos. O teste
compara a candidata, a CNN original e o MLP original nas mesmas amostras
reservadas. O exemplo usa uma pessoa por grupo para ilustrar o comando.
Para medir generalização, reúna várias pessoas em cada grupo.
A avaliação fica em `modelos/avaliacao_cnn_webcam.json`.
Sem CSV, `treinar_cnn.py` salva uma candidata em `modelos/cnn_aprimorada.npz`
e avalia o conjunto sintético do banner e um segundo conjunto com outra semente.
O jogo carrega por padrão `modelos/cnn_faixa_95.npz`. A variante foi
selecionada pela acurácia na validação e acertou 345/360 (95,83%) no conjunto
do banner e 861/900 (95,67%) no segundo conjunto sintético. Para reproduzir:

```powershell
.\.venv\Scripts\python.exe treinar_cnn.py --epocas 6 --faixa-validacao 0.94 0.96 --saida modelos/cnn_faixa_95.npz
```

Use `--modelo modelos/cnn_aprimorada.npz` para jogar com a variante de
99,17% ou `--modelo modelos/cnn.npz` para usar os pesos originais.

## Materiais

- [Documentação de engenharia do checkpoint](docs/Mao_no_Jogo_Documentacao_Engenharia.pdf)
- [Diagrama de arquitetura editável no Draw.io](docs/Mao_no_Jogo_Arquitetura.drawio)
- `docs/Relatorio tecnico - Maycon.docx` e `docs/Guia de integracao - Danilo.docx` documentam a etapa anterior com MLP. A arquitetura atual está descrita neste README.
- `banner/Banner - Mao no Jogo.pdf` é o banner em 90 × 120 cm.
- `banner/Banner editavel - Mao no Jogo.svg` permite atualizar texto e figuras em um editor vetorial. Exporte em 90 × 120 cm após editar.

A equipe realizou o ensaio na bancada e um teste piloto com participantes,
que deram feedback qualitativo positivo. A próxima etapa é registrar
métricas com usuários, incluindo taxa de comandos aceitos e latência
da captura até a ação no jogo.

## Título e resumo para o cadastro

Um dos objetivos do projeto é ampliar o acesso a jogos digitais com uma
alternativa por gestos para quem tem dificuldade para jogar com teclado e
mouse. Avaliar o conforto e a facilidade de uso com esse público é uma
próxima etapa do projeto.

**Mão no Jogo — Controle de um minijogo por gestos com redes neurais**

O Mão no Jogo é um minijogo interativo controlado por gestos capturados pela
webcam. Uma rede neural treinada para o projeto classifica três gestos da mão e
os converte em comandos para mover ou parar o personagem. Durante a exposição,
os visitantes poderão jogar em tempo real e acompanhar na tela qual gesto foi
reconhecido. A aplicação será executada localmente para funcionar sem internet.
