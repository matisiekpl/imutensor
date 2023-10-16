import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = []
labels = []

data_folder = 'data'

classes = {
    'CIRCLES_RIGHT':0,
    'CIRCLES_LEFT':1,
    'TRIANGLE':2,
    'SQUARE':3,
    'FORWARD_BACK':4,
}

for class_folder in os.listdir(data_folder):
    class_folder_path = os.path.join(data_folder, class_folder)
    if not os.path.isdir(class_folder_path):
        continue
    class_label = classes[class_folder]
    for csv_file in os.listdir(class_folder_path):
        if csv_file.endswith('.csv'):
            csv_file_path = os.path.join(class_folder_path, csv_file)
            df = pd.read_csv(csv_file_path, sep=';')
            df = df.iloc[:, :-1]
            # df = df.iloc[:, 0:3]
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

split_ratio = 0.8
split_index = int(len(data) * split_ratio)
x_train, x_test = data[:split_index], data[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]

model = tf.keras.Sequential([
    tf.keras.layers.Input((data.shape[1], data.shape[2])),
    tf.keras.layers.LSTM(22),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(classes.keys()))
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100,validation_data=(x_test,y_test))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

model.summary()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')

model.save('model.h5')