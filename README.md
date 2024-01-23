# Imutensor
LSTM-based classifier written in PyTorch for detecting predefined gestures from IMU sensor.

> This project is developed for Machine Learning course on AGH University of Science and Technology.
> Authors:
> - Mateusz Woźniak
> - Maciej Pawłowski

## Task

This machine learning model takes imu data as a series in `.csv` in format like:
```csv
gyro_x;gyro_y;gyro_z;magnetometer_x;magnetometer_y;magnetometer_z;accelerometer_x;accelerometer_y;accelerometer_z
0.48433977;-0.28244883;1.5225816;-23.34;3.36;-42.78;-2.6527755;-0.9696517;9.857227
0.2222786;-0.95989835;1.2843442;-23.519999;3.1799998;-43.32;-0.4141969;-0.12689269;11.772589
0.61995184;-0.38385245;2.3576343;-24.66;3.8999999;-42.899998;-0.7062895;-1.733402;15.330672
``` 
and classifies it into one from following classes:
```python
classes = {
    'CIRCLES_RIGHT': 0,
    'CIRCLES_LEFT': 1,
    'TRIANGLE': 2,
    'SQUARE': 3,
    'FORWARD_BACK': 4,
}
```

## Dataset collection
We developed mobile app for Android in Kotlin that grabs data from imu sensor and uploads it into Golang microservice which archives files.

![screenshot.jpg](screenshot.jpg)

## Model
We are using LSTM layer with two Linear transformations. Recurrent neural network is used, because gesture-classification task is time-invariant (`.csv` files has diffrent number of timesteps).
```python
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
```

## Quality
We are able to achieve `~95%` accuracy on validation dataset using gyroscope-only features.

## Inference
Model is trained inside `docker build` and saved into `model.pt` weights files. `serve.py` file launches `flask` HTTP server which exposes `POST /inferece` endpoint on port `3199`.

## Running
```bash
docker build -t imutensor .
docker run -p 4199:4199 --name imutensor -d imutensor
```
