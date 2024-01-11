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

DEVICE = 'cpu'

model = Net(3).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()

app = flask.Flask(__name__)


@app.route('/inference', methods=['POST'])
def interference():
    d = request.data.decode('utf-8')
    print(d)
    df = pd.read_csv(StringIO(d), sep=';', skiprows=1)
    df = df.iloc[:, :-1]
    df = df.iloc[:, feature_filter[0]:feature_filter[1]]
    df = (df-df.min())/(df.max()-df.min())
    data = torch.Tensor(df.to_numpy()).unsqueeze(0).to(device)
    output = model(data)
    output = output.cpu().detach().numpy()
    print(output)
    return json.dumps(output.tolist())


cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
log = logging.getLogger('werkzeug')
log.disabled = True
print('Launching inference server...')
app.run(host='0.0.0.0', port=4199)
