import pickle
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
from Metrics import Metrics
from Dataset import Dataset
from Lstm import LSTM
import DataProcessing
import DataToClip
from Trainer import Trainer
import util
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import Configuration
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

cs = ConfigStore.instance()
cs.store(name='execution_config', node=Configuration)

@hydra.main(config_path="conf", config_name='config.yaml', version_base='1.1')
def main(cfg: Configuration):
    if cfg.code_flow_config.do_data_to_clip:
        DataToClip.run(cfg=cfg)
    if cfg.code_flow_config.do_data_processing:
        print("Processing data...")
        DataProcessing.run(cfg=cfg)
    trainer = Trainer(cfg=cfg)
    if cfg.code_flow_config.do_training:
        print("Starting to train model...")
        model = trainer.train(cfg=cfg)
    else:
        pass
        print("Loading model...")
        model = torch.load(cfg.file_path_config.model_path)

    print("Getting prediction from the validation set...")
    labels, predictions, pure_predictions = trainer.get_prediction_from_model(model)
    metrics = Metrics(labels, predictions, pure_predictions, util.lb)
    metrics.classification_report_print()


    print("Finished")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
