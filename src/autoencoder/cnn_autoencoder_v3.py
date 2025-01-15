# a treia incercare facuta,
# aici am un CN autoencoder cu dilatare

# am schimbat arhitectura retelei pentru autoencoder
# de asemenea dataset-ul reprezinta slide-uri de 12 saptamani consecutive (60 de zile)
# fiecare slide fiind decalat cu cate o saptamana (1-12) - (2-13) etc...

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
def get_set_by_years(years):
    return pd.concat([df.loc[df['Date'].dt.year == x] for x in years])

test_set = get_set_by_years([2024])
train_set = get_set_by_years([2022,2023])

# HIPERPARAMETRI

SIZE_INPUT = 12 # 12 saptamani de trading
NUM_STD_DEV = 1

def create_set(dfs):

    return_res = list()
    for x in range(len(dfs) - SIZE_INPUT + 1):
        return_res.append(pd.concat([dfs[y] for y in range(x,x+SIZE_INPUT)]))

    return return_res

# reference
# https://stackoverflow.com/questions/71646721/how-to-split-a-dataframe-by-week-on-a-particular-starting-weekday-e-g-thursday
def get_dataframe_for_year_in_weeks(df, year):

    df_year = df[df['Date'].dt.year == year]
    result_before = [x for _,x in df.groupby(df_year['Date'].dt.to_period('W'))]
    result_after = []

    for elem in result_before:
        if (elem.shape == (5,8)):
            result_after.append(elem)

    # normalizez datele inainte sa fac sliding window
    aux = pd.concat([result_after[y] for y in range(len(result_after))])
    mean = aux['Volume'].mean()
    std = aux['Volume'].std()

    for x in result_after:
        x['Volume'] = (x['Volume'] - mean) / std

    return create_set(result_after)

# metoda care intoarce o multime de antrenare intr-un sir continuu de date
# (31, 60, 1) -> ()
def revert_sequences(values):
    print(values.shape)
    bins = values.shape[0]
    step = SIZE_INPUT
    output = []

    for i in range(0,bins,step):
        for x in values[i]:
            output.append(x)
    print(len(output))

    remainder = (bins % SIZE_INPUT) * 5

    print(len(values[bins - 1][-remainder:]))

    for x in values[bins - 1][-remainder:]:
        output.append(x)

    return np.array(output)


dfs_2022 = get_dataframe_for_year_in_weeks(df, 2022)
dfs_2023 = get_dataframe_for_year_in_weeks(df, 2023)

train_volume = np.array([x['Volume'] for x in dfs_2022])
test_volume = np.array([x['Volume'] for x in dfs_2023])

x_train = train_volume
x_train.resize((x_train.shape[0], x_train.shape[1], 1))

x_test = test_volume
x_test.resize((x_test.shape[0], x_test.shape[1], 1))

print(x_test.shape)

#model cnn autoencoder
#padding = same adauga zero-padding a.i. dimensiunea seriei de timp ramane neafectata de convolutie
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1],x_train.shape[2])),
        layers.Conv1D(filters=20,kernel_size=5, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=5),
        layers.Conv1D(filters=20,kernel_size=4, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=2),
        layers.Conv1DTranspose(filters=20,kernel_size=4, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=2),
        layers.Conv1DTranspose(filters=20,kernel_size=5, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=5),
        layers.Conv1DTranspose(filters=1, kernel_size = 1, padding = "same")
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")


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

x_train_pred = model.predict(x_train)
x_test_pred = model.predict(x_test)

x_test_pred = revert_sequences(x_test_pred)

plt.figure()
plt.plot(revert_sequences(x_test))
plt.plot(x_test_pred)
plt.show()

absolute_difference = np.abs(revert_sequences(x_test) - x_test_pred)

plt.plot(absolute_difference)
plt.show()

mean_absdif = np.mean(absolute_difference)
std_absdif = np.std(absolute_difference)

anomaly_points = np.where(absolute_difference >= mean_absdif + NUM_STD_DEV*std_absdif)
print(anomaly_points)
anomalous_points = revert_sequences(test_volume)[anomaly_points[0]]
print(anomalous_points)

if not os.path.exists('./src/autoencoder/reports/'):
    os.makedirs('./src/autoencoder/reports/')

first_day_test = test_set.index[0]

#test_set.loc[[x + first_day_test for x in anomaly_points[0]]].to_html('./src/autoencoder/reports/CN_auto_v3.html')

plt.figure()
plt.plot(revert_sequences(test_volume))
plt.plot(anomaly_points[0], revert_sequences(test_volume)[anomaly_points[0]], 'ro')
plt.show()