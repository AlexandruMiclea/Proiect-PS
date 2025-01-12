# TODO modifica descrierea asta
# TODO modifica peste tot la mse
# TODO gaseste o metoda de a detecta puncte de anomalii si a le plota direct doar pe acelea
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

def get_set_by_years(years):
    return pd.concat([df.loc[df['Date'].dt.year == x] for x in years])

test_set = get_set_by_years([2022,2023,2024])
train_set = get_set_by_years([x for x in range(2017, 2022)])

#print(test_set)
#print(train_set)

# ce vreau sa fac
# iau volumul actiunilor in anii 2021 2022 si 2023 si antrenez un model autoencoder pe seria asta
# folosesc modelul pentru a detecta anomalii in 2024

# HIPERPARAMETRI

SIZE_INPUT = 90
NUM_STD_DEV = 3

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
# TODO modifica arhitectura sa fie cu temporal autoencoders
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1],x_train.shape[2])),
        layers.Conv1D(filters=5,kernel_size=5, strides = 1, padding = "same", activation="relu", dilation_rate=5),
        #layers.Dropout(rate = 0.1),
        layers.Conv1D(filters=4,kernel_size=4, strides = 1, padding = "same", activation="relu", dilation_rate=4),
        layers.Conv1DTranspose(filters=4,kernel_size=4, strides = 1, padding = "same", activation="relu", dilation_rate=4),
        #layers.Dropout(rate = 0.1),
        layers.Conv1DTranspose(filters=5,kernel_size=5, strides = 1, padding = "same", activation="relu", dilation_rate=5),
        layers.Conv1DTranspose(filters=1, kernel_size = 1, padding = "same")
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

# TODO create a temporal cnn autoencoder and transformer-based autoencoders

model_log = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
)

plt.plot(model_log.history["loss"], label = "train loss")
plt.plot(model_log.history["val_loss"], label = "validation loss")
plt.show()

# Luam MSE intre predictiile pe train si modelul original
x_train_pred = model.predict(x_train)
train_mse_loss = np.mean((x_train_pred - x_train)**2, axis=1)

plt.hist(train_mse_loss, bins=50)
plt.xlabel("Train MSE loss")
plt.ylabel("No of samples")
plt.legend()
plt.show()

# # Get reconstruction loss threshold (MSE, more sensible to outliers).

# print("Reconstruction error threshold: ", threshold)

# # Checking how the first sequence is learnt
plt.plot(revert_sequences(x_train))
plt.plot(revert_sequences(x_train_pred))
plt.show()

# mse between test predictions and test gt
x_test_pred = model.predict(x_test)
test_mse_loss = np.mean((x_test_pred - x_test) ** 2, axis=1)

plt.plot(revert_sequences(x_test))
plt.plot(revert_sequences(x_test_pred))
plt.show()

threshold = np.mean(train_mse_loss) + NUM_STD_DEV*np.std(train_mse_loss)
anomalies = test_mse_loss > threshold

anomalies = np.where(anomalies)
anomalous_points = test_volume.iloc[anomalies[0]]

if not os.path.exists('./src/autoencoder/reports/'):
    os.makedirs('./src/autoencoder/reports/')

first_day_test = test_set.index[0]
test_set.loc[[x + first_day_test for x in anomalies[0]]].to_html('./src/autoencoder/reports/CN_auto_v2.html')

plt.figure()
plt.plot(test_volume)
plt.plot(anomalous_points, 'ro')
plt.show()
