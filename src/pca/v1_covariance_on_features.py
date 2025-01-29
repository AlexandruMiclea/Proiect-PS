VERSION = "v1"

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

# HIPERPARAMETRI

NUM_STD_DEV = 1

df = pd.read_csv("./datasets/NVidia_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)

if not os.path.exists('./src/pca/plots/'):
    os.makedirs('./src/pca/plots/')

if not os.path.exists('./src/pca/pickles/'):
    os.makedirs('./src/pca/pickles/')

def get_set_by_years(years):

    return pd.concat([df.loc[df['Date'].dt.year == x] for x in years])

test_set = get_set_by_years([2024])
test_set = test_set[['Volume', 'Open', 'High', 'Low', 'Close']]

for x in test_set.keys():
    test_set[x] = (test_set[x] - test_set[x].mean()) / test_set[x].std()

test_set_np = test_set.to_numpy()

cov_matrix = test_set_np @ test_set_np.T / test_set_np.shape[0]

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

# pe scala logaritmica punctul de inflexiune se afla la 4, pe scala normala punctul de inflexiune
# apare la 1, facem PCA pastrand 1 si 4 dimensiuni

sorted_indices = np.argsort(eigvals)[::-1]

for i in [1,2]:

    # iau cei mai relevanti eigenvectors
    best_eigvec = eigvecs[:,sorted_indices[:i]]

    print(test_set.shape)

    # proiectam seria de timp aferenta volumului in subspatiul iD
    test_set_projected = test_set.T @ best_eigvec

    # reproiectam seria de timp in spatiul original
    test_set_rebuilt = (test_set_projected @ best_eigvec.T).T

    plt.figure()
    plt.title(f"Comparatie intre y si y_rec (PCA({i}))")
    plt.plot(test_set_rebuilt['Volume'], label = "Valori obtinute dupa reducerea dimensionalitatii", linestyle = 'dashed')
    plt.plot(test_set['Volume'].keys() - test_set['Volume'].index[0], test_set['Volume'], label = "Valori originale volum (normalizate)")
    plt.savefig(f"./src/pca/plots/{VERSION}_y_vs_y_rec_PCA({i}).svg", format='svg')
    plt.legend()

    absolute_difference = np.abs(test_set['Volume'].to_numpy() - test_set_rebuilt['Volume'].to_numpy())

    # fac media si deviatia standard
    mean_absdif = np.mean(absolute_difference)
    std_absdif = np.std(absolute_difference)

    # threshold-ul peste care se considera ca am anomalie vs normal behaviour
    threshold = mean_absdif + NUM_STD_DEV*std_absdif

    anomaly_points = np.where(absolute_difference >= threshold)
    anomaly_points += test_set['Volume'].index[0]

    with open(f'./src/pca/pickles/{VERSION}_anomalous_points_PCA({i}).pkl', 'wb') as pickle_file:
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