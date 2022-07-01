import numpy as np
from sklearn import metrics
import pickle
import torch
from Dataset import Dataset


class Metrics:

    def __init__(self, ground_truth: np.ndarray, prediction: np.ndarray, pure_pred: np.ndarray, labels: np.ndarray):
        """ Initialization.

        :param ground_truth: The ground truth.
        :param prediction: The binary prediction.
        :param pure_pred: The prediction outputted from the network
        :param labels: The label list.
        """
        self.gt = ground_truth
        self.prediction = prediction
        self.labels = labels
        self.conf_mat = metrics.multilabel_confusion_matrix(ground_truth, prediction)
        self.classification_report = metrics.classification_report(self.gt,
                                                                   self.prediction,
                                                                   target_names=self.labels,
                                                                   labels=range(0, len(self.labels)),
                                                                   output_dict=True,
                                                                   zero_division=0)
        for i_class in range(len(labels)):
            self.classification_report[labels[i_class]]['AP'] = metrics.average_precision_score(self.gt[:, i_class],
                                                                                                pure_pred[:, i_class])

    def classification_report_print(self):
        """ Prints the matrics for each given class.

        """
        i = 0
        print(f'{"":>12}', end='\t')
        print(f'{"precision":>12}', end='\t')
        print(f'{"recall":>12}', end='\t')
        print(f'{"f1-score":>12}', end='\t')
        print(f'{"support":>12}', end='\t')
        print(f'{"AP":>12}', end='\n')
        for genre in self.classification_report:
            if i > len(self.labels) - 1:
                break
            i += 1
            print(f'{genre:>12}', end='\t')
            for metrics in self.classification_report[genre]:
                if metrics == 'support':
                    print(f'{self.classification_report[genre][metrics]:>12}', end='\t')
                else:
                    print(f'{100 * self.classification_report[genre][metrics]:>12.2f}', end='\t')
            print('')

    @staticmethod
    def get_pred_from_model(model_path):
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.backends.cudnn.benchmark = True
            model = torch.load(model_path)
            with open('C:/Users/hugofac.TECGRAF/Documents/MovieDataSet/list_id.pkl', 'rb') as f:
                list_id = pickle.load(f)
            with open('C:/Users/hugofac.TECGRAF/Documents/MovieDataSet/movie_map.pkl', 'rb') as f:
                movie_map = pickle.load(f)

            dataset = Dataset(list_id, movie_map)
            partition = {'train': [], 'validation': []}
            for i, el in enumerate(list_id):
                if i / len(list_id) < 0.8:
                    partition['train'].append(el)
                else:
                    partition['validation'].append(el)

            validation_set = Dataset(partition['validation'], movie_map)
            validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=True,
                                                               num_workers=4)

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

if __name__ == "__main__":
    lb = ["Drama", "Comedy", "Thriller", "Action", "Romance", "Horror", "Crime", "Documentary", "Adventure",
          "Sci-Fi", "Family", "Fantasy", "Mystery", "Biography", "Animation", "History", "Music", "War", "Sport",
          "Musical", "Western"]

    #gt = np.load("C:/Users/hugofac.TECGRAF/Documents/MovieDataSet/labels.npy")
    #pred = np.load("C:/Users/hugofac.TECGRAF/Documents/MovieDataSet/predictions.npy")

    #gt, pred, pure_pred = MAP.get_pred_from_model(model_path='C:/Users/hugofac.TECGRAF/Documents/MovieDataSet/LSTM_Models/Model_BCEWithLogitsLoss_2022_06_02-133046/model_sigmoidnormalization_600_16_128')
    gt, pred, pure_pred = Metrics.get_pred_from_model(model_path='C:/Users/hugofac.TECGRAF/Documents/MovieDataSet/LSTM_Models/WeightedModel_Bidirectional_Maxpooling_BCELoss_2022_06_21-164728/model_sigmoidnormalization_600_16_128')

    aux = Metrics(gt, pred, lb)

    AP = []
    for i_class in range(len(gt[0])):
        AP.append(metrics.average_precision_score(gt[:, i_class], pure_pred[:, i_class]))

    print(aux.conf_mat)
    aux.classification_report_print(AP)