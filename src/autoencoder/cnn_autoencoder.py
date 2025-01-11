# prima incercare facuta,
# aici este un CNN autoencoder, care reuseste sa extraga fluctuatii in volumul
# tranzactionat pe un indice bursier

# anomalia detectata in cazul de fata este data de earnings report-ul aferent Q1 2023
# https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-first-quarter-fiscal-2023

import os
import numpy as np
import torch
import re
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras.callbacks
from keras import layers
from keras import backend as K
from matplotlib import pyplot as plt
from show_dataset import *

print("Keras backend: ", K.backend())
print("CUDA for Torch: ", torch.cuda.is_available())

df = pd.read_csv("./datasets/NVidia_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)

# mi-a luat 2 ore sa fac asta...
#test_set = pd.concat([df.loc[df['Date'].dt.year == 2023], df[df['Date'].dt.year == 2024]])
#train_set = pd.concat([df.loc[df['Date'].dt.year == 2021], df.loc[df['Date'].dt.year == 2022]])

test_set = pd.concat([df.loc[df['Date'].dt.year == 2023]])
train_set = pd.concat([df.loc[df['Date'].dt.year == 2021], df.loc[df['Date'].dt.year == 2022]])

#print(test_set)
#print(train_set)

# ce vreau sa fac
# iau volumul actiunilor in anii 2021 2022 si 2023 si antrenez un model autoencoder pe seria asta
# folosesc modelul pentru a detecta anomalii in 2024

# HIPERPARAMETRI

SIZE_INPUT = 90 # 3 luni

#show_dataset_info(dataset)

# prepare the training data for volume anomaly detection

train_volume = train_set['Volume']
test_volume = test_set['Volume']

train_volume_mean = train_volume.mean()
train_volume_stddev = train_volume.std()
train_volume_normalized = (train_volume - train_volume_mean) / train_volume_stddev

test_volume_mean = test_volume.mean()
test_volume_stddev = test_volume.std()
test_volume_normalized = (test_volume - test_volume_mean) / test_volume_stddev


def create_sequences(values, time_steps=SIZE_INPUT):
    output = []
    orig_size = len(values)
    for i in range(orig_size - time_steps + 1):
        output.append(values[i:i+time_steps])
    return np.stack(output)


def revert_sequences(values):
    print(values.shape)
    bins = values.shape[0]
    step = values.shape[1]
    output = []

    for i in range(0,bins,step):
        for x in values[i]:
            output.append(x)
    print(len(output))

    remainder = (bins - 1) % step

    print(remainder)

    print(len(values[bins - 1][-remainder:]))

    for x in values[bins - 1][-remainder:]:
        output.append(x)

    return np.array(output)

x_train = create_sequences(train_volume_normalized.values)
x_train.resize((x_train.shape[0], x_train.shape[1], 1))

x_test = create_sequences(test_volume_normalized)
x_test.resize((x_test.shape[0], x_test.shape[1],1))

# model cnn autoencoder
# padding = same adauga zero-padding a.i. dimensiunea seriei de timp dupa convolutie ramane identica
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1],x_train.shape[2])),
        layers.Conv1D(filters=32,kernel_size=7, strides = 2, padding = "same", activation="relu"),
        layers.Dropout(rate = 0.2),
        layers.Conv1D(filters=12,kernel_size=7, strides = 5, padding = "same", activation="relu"),
        layers.Conv1DTranspose(filters=12,kernel_size=7, strides = 5, padding = "same", activation="relu"),
        layers.Dropout(rate = 0.2),
        layers.Conv1DTranspose(filters=32,kernel_size=7, strides = 2, padding = "same", activation="relu"),
        layers.Conv1DTranspose(filters=1, kernel_size = 1, padding = "same")
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

# TODO create a temporal cnn autoencoder and transformer-based autoencoders

model_log = model.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
    ],
)

plt.plot(model_log.history["loss"], label = "train loss")
plt.plot(model_log.history["val_loss"], label = "validation loss")
plt.show()

# Luam MAE intre predictiile pe train si modelul original
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.legend()
plt.show()

# # Get reconstruction loss threshold (MSE, more sensible to outliers).
threshold = np.max(train_mae_loss)
# print("Reconstruction error threshold: ", threshold)

# # Checking how the first sequence is learnt
plt.plot(revert_sequences(x_train))
plt.plot(revert_sequences(x_train_pred))
plt.show()

x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

plt.plot(revert_sequences(x_test))
plt.plot(revert_sequences(x_test_pred))
plt.show()

# print(volume[:INTERVAL_ZILE].std())

anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

anomalous_data_indices = []
for data_idx in range(SIZE_INPUT - 1, len(test_volume_normalized) - SIZE_INPUT + 1):
    if np.all(anomalies[data_idx - SIZE_INPUT + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

print(anomalous_data_indices)

anomalous_points = test_volume.iloc[anomalous_data_indices]

print(test_set)

if not os.path.exists('./src/autoencoder/reports/'):
    os.makedirs('./src/autoencoder/reports/')

# TODO get first index of test dataset

test_set.loc[[x + 6026 for x in anomalous_data_indices]].to_html('./src/autoencoder/reports/CN_auto_v1.html')


plt.figure()
plt.plot(test_volume)
plt.plot(anomalous_points, color = 'r')
plt.show()
