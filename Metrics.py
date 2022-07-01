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
        """ Prints the metrics for each given class.

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
