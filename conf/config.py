from dataclasses import dataclass


@dataclass
class code_flow_config:
  do_data_to_clip: bool
  do_data_processing: bool
  do_training: bool

@dataclass
class model_config:
    model_dir: str
    n_epochs: int
    input_size: int
    sequence_length: int
    batch_size: int
    hidden_size: int
    num_layers: int
    train_split_proportion: float
    class_weights_crossentropy: bool
    patience: int
    learning_rate: float
    learning_decay: int
    learning_decay_ratio: int


@dataclass
class seed_config:
    SEED: int


@dataclass
class file_path_config:
    trailer_dir: str
    encoded_trailer_tensor_dir: str
    processed_tensor_dir: str
    list_id_path: str
    movie_map_path: str
    meta_data_dir: str
    base_dir: str
    model_path: str
    yaml_path: str


@dataclass
class clip_encoder_config:
  batch_size: int


@dataclass
class data_processing_config:
  patch_size: int
  number_patches: int


@dataclass
class Configuration:
    model_config: model_config
    seed_config: seed_config
    file_path_config: file_path_config
    clip_encoder_config: clip_encoder_config
    data_processing_config: data_processing_config
    code_flow_config: code_flow_config

