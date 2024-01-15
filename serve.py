import torch
import flask
from flask import request
from io import StringIO
import json
from model import Net
import pandas as pd
from config import feature_filter, device
import logging
import sys
import numpy as np

DEVICE = 'cpu'

model = torch.load('model.pt').to(DEVICE)

app = flask.Flask(__name__)


@app.route('/inference', methods=['POST'])
def interference():
    d = request.data.decode('utf-8')
    print(d)
    df = pd.read_csv(StringIO(d), sep=';', skiprows=1)
    df = df.iloc[:, :-1]
    df = df.iloc[:, feature_filter[0]:feature_filter[1]]
    df = (df-df.min())/(df.max()-df.min())
    data_array = df.to_numpy()
    result_array = np.zeros((100, 9))
    result_array[:data_array.shape[0], :] = data_array
    data = torch.Tensor(result_array).unsqueeze(0).to(device)
    output = model(data)
    _, predicted = torch.max(output.cpu().detach(), 1)
    output = output.cpu().detach().numpy()
    print(output)
    print(predicted)
    return json.dumps(output.tolist())


cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
log = logging.getLogger('werkzeug')
log.disabled = True
print('Inference server listening...')
app.run(host='0.0.0.0', port=4199)
