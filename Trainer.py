import pickle
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import shutil
from Metrics import Metrics
from Dataset import Dataset
from Lstm import LSTM
import DataProcessing
from hydra.core.config_store import ConfigStore
from conf.config import Configuration
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

class Trainer:

    def __init__(self, cfg: Configuration):
        """ Initialization.

        :param cfg: The .yaml configuration setted.
        """
        self.writer = SummaryWriter()
        self.model = []
        self.validation_set = []

        torch.manual_seed(cfg.seed_config.SEED)

        with open(cfg.file_path_config.list_id_path, 'rb') as f:
            list_id = pickle.load(f)
        with open(cfg.file_path_config.movie_map_path, 'rb') as f:
            movie_map = pickle.load(f)

        torch.backends.cudnn.benchmark = True

        # Splitting dataset into train and validation
        partition = {'train': [], 'validation': []}
        for i, el in enumerate(list_id):
            if i / len(list_id) < cfg.model_config.train_split_proportion:
                partition['train'].append(el)
            else:
                partition['validation'].append(el)

        labels = movie_map  # Labels

        genres_weight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for movie_id in partition['train']:
            for i in range(len(genres_weight)):
                genres_weight[i] += movie_map[movie_id][i]

        self.genres_weight = genres_weight

        # Generators
        training_set = Dataset(partition['train'], labels, cfg.file_path_config.processed_tensor_dir)
        training_generator = torch.utils.data.DataLoader(training_set,
                                                         batch_size=cfg.model_config.batch_size,
                                                         shuffle=True,
                                                         num_workers=4)
        self.training_generator = training_generator

        validation_set = Dataset(partition['validation'], labels, cfg.file_path_config.processed_tensor_dir)
        validation_generator = torch.utils.data.DataLoader(validation_set,
                                                           batch_size=cfg.model_config.batch_size,
                                                           shuffle=True,
                                                           num_workers=4)
        self.validation_generator = validation_generator

    def __to_tensorboard(self, map_dictionary: dict, epoch: int):
        """ Auxiliary function to save the metrics into a tensorboard.

        :param map_dictionary: Dictionary containing the information of the metrics.
        :param epoch: Which epoch the metrics were generated.
        """
        self.writer.add_scalar('Avg precision', map_dictionary['weighted avg']['precision'], epoch)
        self.writer.add_scalar('Avg recall', map_dictionary['weighted avg']['recall'], epoch)
        self.writer.add_scalar('Avg f1-score', map_dictionary['weighted avg']['f1-score'], epoch)

        print(f'Avg precision [weighted:{map_dictionary["micro avg"]["precision"]:.4f}] '
              f'Avg recall [weighted:{map_dictionary["micro avg"]["recall"]:.4f}] '
              f'Avg f1-score [weighted:{map_dictionary["micro avg"]["f1-score"]:.4f}] ')

    def get_prediction_from_model(self, model):
        """ Generates the prediction of a model given the input values.

        :param model: The given model to get the prediction
        :return: Return the labels (ground truth), the binary predictions and the raw output prediction.
        """
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.backends.cudnn.benchmark = True

            validation_generator = self.validation_generator

            labels = []
            predictions = []
            pure_predictions = []
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                prediction = model(local_batch)
                aux_pred = []
                for pred in prediction:
                    single_pred = []
                    for i in range(len(pred)):
                        if pred[i] >= 0.5:
                            single_pred.append(1)
                        else:
                            single_pred.append(0)
                    aux_pred.append(single_pred)

                labels += local_labels.tolist()
                predictions += aux_pred
                pure_predictions += prediction.tolist()

            labels = np.array(labels)
            pure_predictions = np.array(pure_predictions)
            predictions = np.array(predictions)

        return labels, predictions, pure_predictions

    def train(self, cfg: Configuration):
        """ Generates a model with the parameters found the .yaml file.

        :param cfg:  The .yaml configuration setted.
        :return: Returns the trained model.
        """
        torch.manual_seed(cfg.seed_config.SEED)

        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.backends.cudnn.benchmark = True

            weights = torch.FloatTensor(self.genres_weight).to(device)

            # input_size = 784 # 28x28
            num_classes = len(DataProcessing.lb)
            num_epochs = cfg.model_config.n_epochs
            batch_size = cfg.model_config.batch_size
            learning_rate = cfg.model_config.learning_rate

            input_size = cfg.model_config.input_size
            sequence_length = cfg.model_config.sequence_length
            hidden_size = cfg.model_config.hidden_size
            num_layers = cfg.model_config.num_layers

            model_name = "model_sigmoidnormalization_" + str(num_epochs) + '_' + str(batch_size) + '_' + str(hidden_size)
            clock = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S") + '/'

            model_path = cfg.model_config.model_dir[:] + 'Model_' + clock

            if not os.path.exists(model_path):
                os.mkdir(model_path)

            original = cfg.file_path_config.yaml_path
            target = model_path + "config.yaml"

            shutil.copyfile(original, target)

            tensorboard_log_path = model_path + "rnn_tensorboard_" + clock

            if not os.path.exists(tensorboard_log_path):
                os.mkdir(tensorboard_log_path)

            self.writer = SummaryWriter(tensorboard_log_path)

        # Generators
        training_generator = self.training_generator
        validation_generator = self.validation_generator

        model = LSTM(input_size, hidden_size, num_layers, num_classes, sequence_length).to(device)

        # Loss and optimizer
        if cfg.model_config.class_weights_crossentropy:
            criterion = nn.BCELoss(weight=weights)
        else:
            criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=cfg.model_config.learning_decay,
                                        gamma=cfg.model_config.learning_decay_ratio)

        best_f1 = 0
        patience = cfg.model_config.patience
        patience_count = 0
        # Loop over epochs
        for epoch in range(num_epochs):
            # Training
            size = len(training_generator)
            running_loss = 0.0
            for local_batch, local_labels in training_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                optimizer.zero_grad()
                # Model computations
                # Forward pass
                outputs = model(local_batch)

                loss = criterion(outputs, local_labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            running_loss_val = 0.0
            size_val = len(validation_generator)

            labels = []
            predictions = []
            binary_prediction = []
            for local_val_batch, local_val_labels in validation_generator:
                local_val_batch, local_val_labels = local_val_batch.to(device), local_val_labels.to(device)

                outputs = model(local_val_batch)

                loss_val = criterion(outputs, local_val_labels)

                running_loss_val += loss_val.item()

                aux_pred = []
                for pred in outputs:
                    single_pred = []
                    for i in range(len(pred)):
                        if pred[i] >= 0.5:
                            single_pred.append(1)
                        else:
                            single_pred.append(0)
                    aux_pred.append(single_pred)

                labels += local_val_labels.tolist()
                predictions += aux_pred
                binary_prediction += outputs.tolist()

            labels = np.array(labels)
            binary_prediction = np.array(binary_prediction)
            predictions = np.array(predictions)

            metrics = Metrics(ground_truth=labels, prediction=predictions, pure_pred=binary_prediction,
                              labels=DataProcessing.lb)
            classification_report = metrics.classification_report

            f1_score = classification_report['weighted avg']['f1-score']

            if best_f1 < f1_score:
                best_f1 = f1_score
                torch.save(model, model_path + model_name)
                self.model = model
                patience_count = 0

            print(f'Epoch [{epoch + 1}/{num_epochs}] '
                  f'Loss: {running_loss / size:.4f} '
                  f'Val_Loss: {running_loss_val / size_val:.4f} '
                  f'Learning_rate: {optimizer.param_groups[0]["lr"]}')
            self.writer.add_scalar('training loss', running_loss / size, epoch)
            self.writer.add_scalar('validation loss', running_loss_val / size_val, epoch)
            self.writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
            self.__to_tensorboard(classification_report, epoch)

            scheduler.step()

            if patience_count == patience:
                break

            patience_count += 1

        self.writer.flush()
        self.writer.close()

        return torch.load(model_path + model_name)
