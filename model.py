import torch.nn as nn
import torch


def get_loss(outputs, labels):
    criterion = nn.BCELoss()
    loss = criterion(outputs, labels)
    return loss


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, inputs: torch.FloatTensor):
        return inputs.reshape(inputs.shape[0], -1)


class COVIDSeq1D(nn.Module):
    def __init__(self, input_shape, encoding_type='onehot'):
        # encoding type = ['onehot', 'discrete']
        super(COVIDSeq1D, self).__init__()
        self._initialize_layers(input_shape)

    def _initialize_layers(self, input_shape):
        self.first_layer = nn.Sequential(
            nn.Conv1d(input_shape, 4, kernel_size=(128,), stride=(8,)),
            nn.ReLU(),
        )

        self.second_layers = nn.Sequential(*[nn.Conv1d(4, 4, kernel_size=(256,), stride=(8, )),
                       nn.ReLU()] * 3)

        self.third_layer = nn.Sequential(
            nn.Conv1d(4, 4, kernel_size=(128,), stride=(8, )),
            nn.ReLU(),
            Reshape(),
            nn.Linear(252, 1),
            nn.Sigmoid()
        )

        self.model = nn.Sequential(
            self.first_layer,
            self.second_layers,
            self.third_layer
        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


class ExtractLSTMTensor(nn.Module):
    def __init__(self):
        super(ExtractLSTMTensor, self).__init__()

    def forward(self, inputs):
        return inputs[0]


class COVIDSeq1DLSTM(COVIDSeq1D):
    def __init__(self, input_shape):
        super(COVIDSeq1DLSTM, self).__init__(input_shape)

    def _initialize_layers(self, input_shape):
        self.first_layer = nn.Sequential(
            nn.Conv1d(input_shape, 4, kernel_size=(128,), stride=(8,)),
            nn.ReLU(),
        )

        self.second_layers = nn.Sequential(*[nn.Conv1d(4, 4, kernel_size=(256,), stride=(8, )),
                                             nn.ReLU()] * 3)

        self.third_layer = nn.Sequential(
            nn.LSTM(628, 512),
            ExtractLSTMTensor(),
            nn.ReLU(),
            Reshape(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

        self.model = nn.Sequential(
            self.first_layer,
            self.second_layers,
            self.third_layer
        )