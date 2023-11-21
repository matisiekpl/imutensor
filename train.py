import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import flask
from flask import request
from io import StringIO
import json

BATCH_SIZE = 32
SPLIT_RATIO = 0.8
EPOCHS = 100
DEVICE = 'cpu'

data = []
labels = []
data_folder = 'data'
classes = {
    'CIRCLES_RIGHT': 0,
    'CIRCLES_LEFT': 1,
    'TRIANGLE': 2,
    'SQUARE': 3,
    'FORWARD_BACK': 4,
}

for class_folder in os.listdir(data_folder):
    class_folder_path = os.path.join(data_folder, class_folder)
    if not os.path.isdir(class_folder_path):
        continue
    class_label = classes[class_folder]
    for csv_file in os.listdir(class_folder_path):
        if csv_file.endswith('.csv'):
            csv_file_path = os.path.join(class_folder_path, csv_file)
            df = pd.read_csv(csv_file_path, sep=';', skiprows=1)
            df = df.iloc[:, :-1]
            df = df.iloc[:, 0:3]
            # df = df.iloc[:, 3:6]
            # df = df.iloc[:, 6:9]
            df = (df-df.min())/(df.max()-df.min())
            data_array = df.to_numpy()
            data.append(data_array)
            labels.append(class_label)

data = np.array(data)
labels = np.array(labels)

permutation = np.random.permutation(len(data))
data = data[permutation]
labels = labels[permutation]

split_index = int(len(data) * SPLIT_RATIO)
x_train, x_test = data[:split_index], data[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
    torch.Tensor(x_train), torch.Tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
    torch.Tensor(x_test), torch.Tensor(y_test)), batch_size=BATCH_SIZE, shuffle=True)


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


model = Net(data.shape[2]).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        y = F.one_hot(y.long(), len(classes.keys())).float()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            model.eval()

            num_correct = 0
            num_samples = 0

            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    output = model(x)
                    _, predicted = torch.max(output.data, 1)
                    num_samples += y.size(0)
                    num_correct += (predicted == y).sum().item()

            model.train()

            accuracy = num_correct / num_samples
            print(
                f'Epoch {epoch+1}, Batch {batch_idx}, Loss {loss.item()}, Accuracy {accuracy}')

print('Launching interference server...')
app = flask.Flask(__name__)


@app.route('/interference', methods=['POST'])
def interference():
    d = request.data.decode('utf-8')
    print(d)
    df = pd.read_csv(StringIO(d), sep=';', skiprows=1)
    df = df.iloc[:, :-1]
    df = df.iloc[:, 0:3]
    # df = df.iloc[:, 3:6]
    # df = df.iloc[:, 6:9]
    df = (df-df.min())/(df.max()-df.min())
    data = torch.Tensor([df.to_numpy()])
    data = data.to(DEVICE)
    output = model(data)
    output = output.cpu().detach().numpy()
    return json.dumps(output.tolist())


app.run(host='0.0.0.0', port=3199)
