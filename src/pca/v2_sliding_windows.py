VERSION = "v2"

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

# HIPERPARAMETRI

SIZE_INPUT = 12
NUM_STD_DEV = 1

df = pd.read_csv("./datasets/NVidia_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)

if not os.path.exists('./src/autoencoder/plots/'):
    os.makedirs('./src/autoencoder/plots/')

if not os.path.exists('./src/autoencoder/pickles/'):
    os.makedirs('./src/autoencoder/pickles/')

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
test_set = test_set[['Volume', 'Open', 'High', 'Low', 'Close']]

for x in test_set.keys():
    test_set[x] = (test_set[x] - test_set[x].mean()) / test_set[x].std()

test_set_np = test_set.to_numpy()

cov_matrix = np.cov(test_set_np)

# folosesc eigh deoarece matricea de covarianta este real simetrica
eigvals, eigvecs = np.linalg.eigh(cov_matrix)

plt.figure()
plt.title('Scree diagram')
plt.plot(np.sort(eigvals)[::-1], label = 'Eigvals (sorted desc)')
plt.legend()
plt.savefig(f"./src/pca/plots/{VERSION}_scree.svg", format='svg')

plt.figure()
plt.title('Scree diagram - log scale')
plt.plot(np.sort(eigvals)[::-1], label = 'Eigvals (sorted desc)')
plt.legend()
plt.yscale('log')
plt.savefig(f"./src/pca/plots/{VERSION}_scree_log.svg", format='svg')

# pe scala logaritmica punctul de inflexiune de afla la 4, pe scala normala punctul de inflexiune
# apare la 1, facem PCA pastrand 1 si 4 dimensiuni

sorted_indices = np.argsort(eigvals)[::-1]

for i in [1,4]:

    # iau cei mai relevanti eigenvectors
    best_eigvec = eigvecs[:,sorted_indices[:i]]

    # proiectam seria de timp aferenta volumului in subspatiul iD
    test_set_vol_projected = test_set['Volume'] @ best_eigvec

    # reproiectam seria de timp in spatiul original
    test_set_vol_rebuilt = test_set_vol_projected * best_eigvec
    test_set_vol_rebuilt = np.sum(test_set_vol_rebuilt, axis = 1)

    plt.figure()
    plt.title(f"Comparatie intre y si y_rec (PCA({i}))")
    plt.plot(test_set_vol_rebuilt, label = "Valori obtinute dupa reducerea dimensionalitatii", linestyle = 'dashed')
    plt.plot(test_set['Volume'].keys() - test_set['Volume'].index[0], test_set['Volume'], label = "Valori originale volum (normalizate)")
    plt.legend()

    absolute_difference = np.abs(test_set['Volume'] - test_set_vol_rebuilt)

    # fac media si deviatia standard
    mean_absdif = np.mean(absolute_difference)
    std_absdif = np.std(absolute_difference)

    # threshold-ul peste care se considera ca am anomalie vs zgomot
    threshold = mean_absdif + NUM_STD_DEV*std_absdif

    anomaly_points = np.where(absolute_difference >= threshold)
    anomaly_points += test_set['Volume'].index[0]

    with open(f'./src/pca/pickles/anomalous_points.pkl', 'wb') as pickle_file:
        pickle.dump(anomaly_points, pickle_file)

    plt.figure()
    plt.title(f'Diferenta in modul intre y si y_rec (PCA({i}))')
    plt.plot(test_set['Volume'].keys() - test_set['Volume'].index[0], absolute_difference, label = 'Diferenta in modul')
    plt.hlines(mean_absdif, 0, absolute_difference.shape[0], label = 'Media diferentelor', linestyles='solid', color = 'blue')
    plt.hlines(threshold, 0, absolute_difference.shape[0], label = f'Threshold (medie + {NUM_STD_DEV}*std)', linestyles='dashed', color = 'black')
    plt.savefig(f"./src/pca/plots/{VERSION}_diferenta_modul_PCA({i}).svg", format='svg')
    plt.legend()

    plt.figure()
    plt.title(f'Anomalii detectate (PCA({i}))')
    plt.plot(test_set['Volume'], label = 'Volumul observat intr-o zi')
    plt.plot(test_set['Volume'].loc[anomaly_points[0]], 'ro', label = 'Punct de anomalie')
    plt.legend()
    plt.savefig(f"./src/pca/plots/{VERSION}_anomalii_detectate_PCA({i}).svg", format='svg')

plt.show()