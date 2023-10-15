
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

filename = 'data/SQUARE/bttqjzgwkz.csv'

df = pd.read_csv(filename, sep=';')
df = df.iloc[:, :-1]
df = (df-df.min())/(df.max()-df.min())
data_array = df.to_numpy()

model = tf.keras.models.load_model('model.h5')

prediction = model.predict(np.array([data_array]))
print(prediction.argmax())