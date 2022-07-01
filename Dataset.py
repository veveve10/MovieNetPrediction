import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, list_IDs: list, labels: dict, path: str):
        """ Initialization.

        :param list_IDs: List of id.
        :param labels: Dictionary where the id is the key and the label is the result.
        :param path: The path to the directory the data is stored
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.path = path

    def __len__(self) -> int:
        """ Denotes the total number of samples.

        :return: Size of the dataset
        """
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        """ Generates one sample of data.

        :param index: The id of the sample to be generated
        :return: Returns the tensor for the trailer if the given id and the label for it.
        """

        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        path = self.path + ID + '.npy'  # ex: r'D:/MovieDataset/output_tensor_new_padded_augmented/' + ID + '.npy'
        X = torch.from_numpy(np.load(path)).float()
        y = torch.Tensor(self.labels[ID])

        return X, y
