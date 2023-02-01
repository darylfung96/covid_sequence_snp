import torch.nn as nn
import torch
from prefect import task


class COVIDSeq1D(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(COVIDSeq1D, self).__init__()
        self.first_layer = nn.Conv1d(input_shape, 4, kernel_size=(128,), stride=(8,))

        self.second_layers = nn.Sequential(*[nn.Conv1d(4, 4, kernel_size=(256,), stride=(8, )),
                       nn.ReLU()] * 3)

        self.third_layer = nn.Conv1d(4, 4, kernel_size=(128,), stride=(8, ))

        self.fc = nn.Linear(252, 1)

    def forward(self, inputs):
        outputs = self.first_layer(inputs)
        outputs = self.second_layers(outputs)
        outputs = self.third_layer(outputs)
        outputs = self.fc(outputs.reshape(inputs.shape[0], -1))
        return torch.sigmoid(outputs)


@task
def get_loss(outputs, labels):
    criterion = nn.BCELoss()
    loss = criterion(outputs, labels)
    return loss