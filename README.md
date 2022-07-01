# MovienetPrediction



Este projeto é focado na construção e treinamento de um modelo de machine learning voltado para classificação de trailers de filmes. Esse problema acaba sendo um problema de multi labels uma vez que um filme pode ter mais de um gênero. O código atual serve para qualquer usuário que tenha a intenção de realizar o mesmo tipo de problema mas permitindo que o mesmo só precise alterar o modelo utilizado, neste caso, a classe LSTM.
O dataset escolido é o da MovieNet (http://movienet.site/) e para poder adquirir e rodar os algoritmos necessario será preciso baixar os seguintes dados que não estão disponibilizados neste git uma vez que possuem tamanho superior a 100GB:
1. Movie per-shot keyframes (240P) (tamanho: 161GB, depois de dar unzip: 161GB)
2. Meta (tamanho: 537MB, depois de dar unzip: 2.3GB)

O projeto se resume na seguinte arquitetura:
![alt text](https://github.com/veveve10/MovieNetPrediction/blob/main/img/Diagram.PNG?raw=true "Diagram")

O arquivo DataToClip.py tem a função de transformar os frames do filmes em um vetor [N, 512] onde N é o número de frames do filme. Para isso, ele utiliza o encoder de imagens do CLIP (https://openai.com/blog/clip/) e o código para essa rede pode ser encontrado em https://github.com/openai/CLIP. Os arquivos serão salvos como numpy arrays.


Em seguida, o DataProcessing irá pegar esse dado e, dado um valor escolhido pelo usuário, irá colocar cada array com essa dimensão ao invés de N, assim tornando o dado uniforme. O DataProcessing também gera dois arquivos auxiliares com a lista dos nomes dos arquivos que servem como identificadores para o segundo arquivo que contem um dicionário cuja chave é esse identificador e o valor armazenado é a lista com seus genero em formato de one-hot encoding.

O usuário pode passar os parâmetros necessários para o código através de um arquivo .yaml que está na pasta "conf" deste projeto. Todos os parametros passado pelo usuários se encontram nesse mesmo arquivo para fácilitar a reprodutibilidade do código. Maior documentação sobre ele se encontra no arquivo .md dentro da pasta "conf".


A classe Trainer tem o papel de, usando a classe LSTM, gerar um modelo com a parametrização passada pelo arquivo .yaml e treina-lo com os parâmetros também presentes no mesmo arquivo. Para que possa se ver o histórico do treinamento, é salvo um tensorboard que permite a visualização e acompanhamento as métricas do treino. Na mesma página é salvo o .yaml usado para o código para que esse treino possa ser replicado caso alguma alteração seja feita com o dado. É possivel extrair as métricas do modelo usando os dados de validação.


A classe Metrics calcula as métricas de um dado modelo e dado de entrada, assim não é necessario treinar todo um novo modelo para poder se extrair essas métricas.


## Resultados

Estarei comparando os resultados de alguma redes geradas e disponibilizadas a parte com os resultado do artigo [A Unified Framework for Shot Type Classification Based on Subject Centric Lens](https://arxiv.org/pdf/2008.03548.pdf). O resultado do artigo vem de uma classificação das imagens presentes em todo o dataset, contendo também mais classes de filmes do que as presentes nos trailers, como por exemplo animação. Isso poderá ser visto nos resultados das redes gerados nos modelos treinados.
Abaixo os resultados publicados (Resnet) contra os resultados pelo modelo LSTM Bidirectional_Maxpooling gerado pelo o programa e disponivel na pasta Models:

| Resnet      | precision@0.5 | recall@0.5 | AP    | LSTM Bidirectional_Maxpooling | precision@0.5 | recall@0.5 | AP    |
|-------------|---------------|------------|-------|-------------------------------|---------------|------------|-------|
| Drama       | 71.16         | 79.42      | 79.95 | Drama                         | 77.67         | 79.97      | 80.31 |
| Comedy      | 68.61         | 48.65      | 68.81 | Comedy                        | 57.09         | 54.84      | 60.30 |
| Thriller    | 64.98         | 14.50      | 49.80 | Thriller                      | 61.49         | 48.79      | 58.87 |
| Action      | 73.96         | 22.21      | 54.60 | Action                        | 67.18         | 54.72      | 66.12 |
| Romance     | 71.93         | 14.02      | 49.27 | Romance                       | 46.79         | 34.27      | 46.42 |
| Horror      | 70.03         | 8.76       | 35.51 | Horror                        | 50.00         | 22.55      | 31.53 |
| Crime       | 74.12         | 39.30      | 49.25 | Crime                         | 54.55         | 36.68      | 48.35 |
| Documentary | 85.49         | 4.79       | 21.03 | Documentary                   | 0.00          | 0.00       | 1.06  |
| Adventure   | 75.24         | 24.72      | 53.06 | Adventure                     | 59.75         | 42.60      | 53.74 |
| Sci-Fi      | 81.35         | 14.51      | 44.14 | Sci-Fi                        | 61.07         | 50.00      | 58.08 |
| Family      | 82.55         | 27.11      | 52.19 | Family                        | 50.00         | 20.93      | 27.78 |
| Fantasy     | 69.83         | 13.51      | 39.12 | Fantasy                       | 52.94         | 28.35      | 39.62 |
| Mystery     | 76.42         | 7.76       | 39.70 | Mystery                       | 42.50         | 25.76      | 30.74 |
| Biography   | 100.00        | 0.04       | 09.13 | Biography                     | 46.55         | 26.73      | 34.45 |
| Animation   | 93.16         | 74.09      | 86.45 | Animation                     | 0.00          | 0.00       | nan   |
| History     | 82.90         | 12.52      | 34.41 | History                       | 50.00         | 27.08      | 36.49 |
| Music       | 89.04         | 27.24      | 47.13 | Music                         | 100.00        | 7.69       | 19.70 |
| War         | 86.27         | 12.80      | 34.41 | War                           | 54.55         | 25.71      | 40.09 |
| Sport       | 94.97         | 21.99      | 39.59 | Sport                         | 50.00         | 13.33      | 34.92 |
| Musical     | 73.58         | 4.45       | 22.88 | Musical                       | 85.71         | 42.86      | 56.77 |
| Western     | 88.89         | 51.93      | 73.99 | Western                       | 66.67         | 8.33       | 22.02 |


Podemos ver que no geral, a precisão da Resnet é superior a do modelo treinado, porém existe uma troca muito grande com o recall apresentado, enquanto o modelo mostrado busca um balanço entre as métricas melhor.













