# am schimbat arhitectura retelei pentru autoencoder
# de asemenea dataset-ul reprezinta slide-uri de 12 saptamani consecutive (60 de zile)
# fiecare slide fiind decalat cu cate o saptamana (1-12) - (2-13) etc...

import os
import numpy as np
import torch
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras.callbacks
from keras import layers
from keras import backend as K
from matplotlib import pyplot as plt
import pickle

# HIPERPARAMETRI

SIZE_INPUT = 12 # 12 saptamani de trading
NUM_STD_DEV = 1

df = pd.read_csv("./datasets/NVidia_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)

if not os.path.exists('./src/autoencoder/plots/'):
    os.makedirs('./src/autoencoder/plots/')

# mi-a luat 2 ore sa fac asta...
def get_set_by_years(years):

    return pd.concat([df.loc[df['Date'].dt.year == x] for x in years])

def create_set(dfs):

    return_res = list()
    for x in range(len(dfs) - SIZE_INPUT + 1):
        return_res.append(pd.concat([dfs[y] for y in range(x,x+SIZE_INPUT)]))

    return return_res

# referinta
# https://stackoverflow.com/questions/71646721/how-to-split-a-dataframe-by-week-on-a-particular-starting-weekday-e-g-thursday
def get_dataframe_in_weeks(df):
    result_before = [x for _,x in df.groupby(df['Date'].dt.to_period('W'))]
    result_after = []

    # filtrez datele sa fie compuse doar din saptamani de trading complete
    for elem in result_before:
        if (elem.shape == (5,8)):
            result_after.append(elem)

    # normalizez datele inainte sa fac sliding window 
    # (astfel media si deviatia standard sunt calculate pentru dataset-ul intreg)
    aux = pd.concat([result_after[y] for y in range(len(result_after))])
    mean = aux['Volume'].mean()
    std = aux['Volume'].std()

    # normalizez fiecare fereastra
    for x in result_after:
        x['Volume'] = (x['Volume'] - mean) / std

    return create_set(result_after)

# metoda care intoarce o multime de antrenare intr-un sir continuu de date
def revert_sequences(values):
    bins = values.shape[0]
    step = SIZE_INPUT
    output = []

    for i in range(0,bins,step):
        for x in values[i]:
            output.append(x)

    remainder = ((bins  - 1) % SIZE_INPUT) * 5

    for x in values[bins - 1][-remainder:]:
        output.append(x)

    return np.array(output)

test_set = get_set_by_years([2024])
train_set = get_set_by_years([x for x in range(2022, 2024)])

dfs_train = get_dataframe_in_weeks(train_set)
dfs_test = get_dataframe_in_weeks(test_set)

# retin prima zi din multimea de dataframes ca sa imi intoarca 
first_day_dfs_test = dfs_test[0].index[0]

train_volume = np.array([x['Volume'] for x in dfs_train])
test_volume = np.array([x['Volume'] for x in dfs_test])

x_train = train_volume
x_train.resize((x_train.shape[0], x_train.shape[1], 1))

x_test = test_volume
x_test.resize((x_test.shape[0], x_test.shape[1], 1))

#model cnn autoencoder
#padding = same adauga zero-padding a.i. dimensiunea seriei de timp ramane neafectata de convolutie

# TODO reteaua asta s-a vrut a fi undercomplete, insa e overcomplete (am 60 de linii si o coloana, filtrele de convolutie nu se aplica bine)
# in mijloc dimensiunea e (60, 20) ceea ce creste dimensiunea fata de input
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1],x_train.shape[2])),
        layers.Conv1D(filters=20,kernel_size=5, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=5),
        layers.Conv1D(filters=20,kernel_size=4, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=4),
        layers.Conv1DTranspose(filters=20,kernel_size=4, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=4),
        layers.Conv1DTranspose(filters=20,kernel_size=5, strides = 1, padding = "same", activation="leaky_relu", dilation_rate=5),
        layers.Conv1DTranspose(filters=1, kernel_size = 1, padding = "same")
    ]
)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

model_log = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
)

plt.figure(figsize=(19.20, 10.80))
plt.title("Statistici antrenare")
plt.plot(model_log.history["loss"], label = "Loss antrenare")
plt.plot(model_log.history["val_loss"], label = "Loss validare")
plt.legend()
plt.yscale('log')
plt.savefig("./src/autoencoder/plots/train_stats.svg", format='svg')

# trec datele pe care vreau sa le prezic 
x_test_pred = model.predict(x_test)
x_test_pred = revert_sequences(x_test_pred)

plt.figure(figsize=(19.20, 10.80))
plt.title("Comparatie intre y si y_pred")
plt.plot(revert_sequences(x_test), label = "Valori prezise", linestyle = 'dashed')
plt.plot(x_test_pred, label = "Valori GT")
plt.legend()
plt.savefig("./src/autoencoder/plots/y_vs_ypred.svg", format='svg')

# fac diferenta in modul pentru a detecta unde anume sunt anomaliile pe seria de timp
absolute_difference = np.abs(revert_sequences(x_test) - x_test_pred)

# fac media si deviatia standard
mean_absdif = np.mean(absolute_difference)
std_absdif = np.std(absolute_difference)

# threshold-ul peste care se considera ca am anomalie vs zgomot
threshold = mean_absdif + NUM_STD_DEV*std_absdif

plt.figure(figsize=(19.20, 10.80))
plt.title('Diferenta in modul intre test si test_pred')
plt.plot(absolute_difference, label = 'Diferenta in modul')
plt.hlines(mean_absdif, 0, absolute_difference.shape[0], label = 'Media diferentelor', linestyles='solid', color = 'blue')
plt.hlines(threshold, 0, absolute_difference.shape[0], label = f'Threshold (medie + {NUM_STD_DEV}*std)', linestyles='dashed', color = 'black')
plt.legend()
plt.savefig("./src/autoencoder/plots/diferenta_modul_autoencoder.svg", format='svg')

# iau indicii pentru punctele considerate a fi anomalii
anomaly_points = np.where(absolute_difference >= threshold)

# convertesc indicii de mai sus (relativi) la indicii care sunt prezenti in dataframe
anomalous_points = []

for val in anomaly_points[0]:

    # big hack incoming
    aux = [x for _,x in df.groupby(test_set['Date'].dt.to_period('W'))]
    result_after = []

    for elem in aux:
        if (elem.shape == (5,8)):
            result_after.append(elem)

    aux = pd.concat([result_after[y] for y in range(len(result_after))])

    anomalous_points.append(aux.index[val])

anomalous_points = np.array(anomalous_points)

# creez directorul pentru punctele cu anomalii
if not os.path.exists('./src/autoencoder/pickles/'):
    os.makedirs('./src/autoencoder/pickles/')

with open(f'./src/autoencoder/pickles/anomalous_points.pkl', 'wb') as pickle_file:
    pickle.dump(anomalous_points, pickle_file)

if not os.path.exists('./src/autoencoder/reports/'):
    os.makedirs('./src/autoencoder/reports/')

# afisez date despre indicele bursier in zilele in care am detectat anomalie
test_set.loc[[x for x in anomalous_points]].to_html('./src/autoencoder/reports/CN_auto_prezentare.html')

plt.figure(figsize=(19.20, 10.80))
plt.title('Anomalii detectate')
plt.plot(test_set['Volume'], label = 'Volumul observat intr-o zi')
plt.plot(anomalous_points, test_set['Volume'].loc[anomalous_points], 'ro', label = 'Punct de anomalie')
plt.legend()
plt.savefig("./src/autoencoder/plots/anomalii_detectate_autoencoder.svg", format='svg')

plt.show()