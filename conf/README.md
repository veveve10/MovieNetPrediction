# YAML File

O arquivo config.yaml deve exitir em par com o config.py. Para poder alterar as variáveis do projeto, deve-se editar o config.yaml. Aqui é a documentação do que serve cada variável.

**code_flow_config**:
  **do_data_to_clip**: booleano que indica se será necessário transformar os dados de imagem em vetor.
  **do_data_processing**: booleano que indica se será necessário uniformizar o tamado do dado gerado.
  **do_training**: booleano que indica se será necessário realizar o treino de um novo modelo ou se o programa só irá carregar um modelo já existente.

**model_config**:
  **model_dir**: string com o caminho para o diretorio onde o modelo ficará, deve terminar com '/'.
  **n_epochs**: inteiro que indica o número de épocas que o modelo irá treinar.
  **input_size**: inteiro que indica o formato do input size, para esse caso deve ser 512.
  **sequence_length**: inteiro que representa quantos frames consecutivos do filme foi escolido para fazer a uniformização.
  **batch_size**: inteiro que indica o tamanho do batch de treino.
  **hidden_size**: inteiro que indica.
  **num_layers**: inteiro que indica o número de camadas LSTM que a rede vai ter.
  **train_split_proportion**: float entre 0-1 que indica quanto % do dado é de treino, a diferença para 1 será a quantidade de dado para validação.
  **class_weights_crossentropy**: booleano que indica se será usado os pesos das classes no calculo da função de perda.
  **patience**: inteiro que indica o número de épocas que o modelo irá aceitar sem ter uma melhora nos resultados.
  **learning_rate**: float que indica qual será o learning rate do modelo. (reconda-se usar 0.001)
  **learning_decay**: inteiro que indica depois de quantas épocas o learning rate irá cair em uma certa razão.
  **learning_decay_ratio**: float que indica a razão em que o learning rate vai decair.
  **max_pooling**: inteiro que representa por quanto a última camada será reduzida.
  **isbiderectional**: booleano que indica se a LSTM será bidirecional ou não.


**seed_config**:
  **SEED**: inteiro para servir como semente randomica para garantir repodutibilidade dos treinos.

**file_path_config**:
  **trailers_dir**: string com o path para o diretorio onde os filmes se encontrará/encontram, precisa terminar com '/\*'.
  **encoded_trailer_tensor_dir**: string com o path para o diretorio onde os filmes encodados pela CLIP se encontrará/encontram, precisa terminar com '/\*'.
  **processed_tensor_dir**: string com o path para o diretorio onde os filmes encodados pela CLIP e foram uniformizados se encontrará/encontram, precisa terminar com '/'.
  **list_id_path**: string com o path para onde o arquivo com a lista dos id se encontrará/encontram, precisa terminar com ".pkl"
  **movie_map_path**: string com o path para onde o arquivo com o dicionario dos filmes e seus gêneros se encontrará/encontram, precisa terminar com ".pkl"
  **meta_data_dir**: string com o path para o diretorio onde o meta dados dos filmes se encontram, precisa terminar com '/'.
  **model_path**: string com o path para o diretorio onde os modelos serão salvos, precisa terminar com '/'.
  **yaml_path**: string com o path deste arquivo.


**clip_encoder_config**:
  **batch_size**: inteiro que indica o batch size de quantos frames de um filme será passado direto para a CLIP.

**data_processing_config**:
  **patch_size**: inteiro que indica o número de frames que será pego de um filme para a uniformização.
  **number_patches**: inteiro que indica o número de vezes que você irá pegar frames de um mesmo filme caso ele possua um número de frames superior ao patch_size.

