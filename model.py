import torch
import torch.nn as nn

classes = {
    'CIRCLES_RIGHT': 0,
    'CIRCLES_LEFT': 1,
    'TRIANGLE': 2,
    'SQUARE': 3,
    'FORWARD_BACK': 4,
}


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, 22, batch_first=True)
        self.fc1 = nn.Linear(22, 32)
        self.fc2 = nn.Linear(32, len(classes.keys()))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1, :, :]
        x = self.fc1(x)
        x = self.fc2(x)
        return x
