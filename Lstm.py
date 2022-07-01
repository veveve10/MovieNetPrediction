import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 seq_length: int,
                 max_pooling: int = 1,
                 bidirectional: bool = False):
        """ Initialization.

        :param input_size: The size of the input. (in this case, the length of the CLIP Encoding, being 512.)
        :param hidden_size: Number of hidden LSTM neurons.
        :param num_layers: Number of LSTM layers.
        :param num_classes: Number of classes.
        :param seq_length: The sequence length of the data. (in this case, the number of frames that was picked to encode)
        :param max_pooling: How many parts the data will be divided into.
        :param bidirectional: If the LSTM is bidirectional or not.
        """
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.LayerNorm = torch.nn.LayerNorm([seq_length, input_size])
        self.max_pooling = torch.nn.MaxPool1d(max_pooling)

        if bidirectional:
            self.fc = nn.Linear(int(2*hidden_size/max_pooling), num_classes)
        else:
            self.fc = nn.Linear(int(hidden_size/max_pooling), num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)

        # tensor of shape (batch_size, seq_length, hidden_size)
        # x: (n, 256, 512)

        # Forward propagate LSTM
        out = self.LayerNorm(x)
        out, _ = self.lstm(out)
        # out: (n, 256, 128)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.max_pooling(out)
        # out: (n, 256, 32) (considering max_pooling equal to 4)

        out = out[:, -1, :]  # out: (n, 32)
        # Decode the hidden state of the last time step

        # out: (n, 32)
        out = self.fc(out)
        out = self.sigmoid(out)
        # out: (n, 21)
        return out
