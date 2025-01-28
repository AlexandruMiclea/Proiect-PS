import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

# HIPERPARAMETRI

VERSION = "v3"
GAMMA = 0.1
COMPONENT_NO = [7]
NUM_STD_DEV = 1

# metoda care imi genereaza matricea kernel rbf pentru setul de date
def rbf_kernel(df, gamma):

    samples = df.shape[0]

    matrix = np.zeros((samples,samples))

    for i in range(samples):
        for j in range(samples):
            matrix[i,j] = np.linalg.norm(df[i] - df[j], ord = 2)**2

    matrix = np.exp(-gamma * matrix)

    return matrix

# metoda care imi centreaza matricea kernel furnizata ca parametru
def center_kernel(matrix):
    N = matrix.shape[0]
    n_1 = np.ones((N,N)) / N

    result_matrix = matrix - matrix @ n_1 - n_1 @ matrix + n_1 @ matrix @ n_1

    return result_matrix

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

# normalizez datele
for x in test_set.keys():
    test_set[x] = (test_set[x] - test_set[x].mean()) / test_set[x].std()

test_set_np = test_set.to_numpy()

# obtin matricea kernel pentru seria mea de timp
matrice_kernel = rbf_kernel(test_set_np, GAMMA)

matrice_centrata = center_kernel(matrice_kernel)

# folosesc eigh deoarece matricea kernel centrata este real simetrica
eigvals, eigvecs = np.linalg.eigh(matrice_centrata)

plt.figure(figsize=(19.20, 10.80))
plt.title(f'Scree diagram (gamma = {GAMMA})')
plt.plot(np.sort(eigvals)[::-1], label = 'Eigvals (sorted desc)')
plt.legend()
plt.savefig(f"./src/pca/plots/{VERSION}_scree_{GAMMA}.svg", format='svg')

sorted_indices = np.argsort(eigvals)[::-1]

for i in COMPONENT_NO:

    # iau cei mai relevanti eigenvectors
    best_eigvec = eigvecs[:,sorted_indices[:i]]

    # proiectam seria de timp aferenta volumului in subspatiul iD
    test_set_projected = test_set.T @ best_eigvec

    # reproiectam seria de timp in spatiul original
    test_set_rebuilt = (test_set_projected @ best_eigvec.T).T

    plt.figure(figsize=(19.20, 10.80))
    plt.title(f"Comparatie intre y si y_rec (PCA({i}))")
    plt.plot(test_set_rebuilt['Volume'], label = "Valori obtinute dupa reducerea dimensionalitatii", linestyle = 'dashed')
    plt.plot(test_set['Volume'].keys() - test_set['Volume'].index[0], test_set['Volume'], label = "Valori originale volum (normalizate)")
    plt.savefig(f"./src/pca/plots/{VERSION}_y_vs_y_rec_PCA({i})_gamma_{GAMMA}.svg", format='svg')
    plt.legend()

    absolute_difference = np.abs(test_set['Volume'].to_numpy() - test_set_rebuilt['Volume'].to_numpy())

    # fac media si deviatia standard
    mean_absdif = np.mean(absolute_difference)
    std_absdif = np.std(absolute_difference)

    # threshold-ul peste care se considera ca am anomalie vs normal behaviour
    threshold = mean_absdif + NUM_STD_DEV*std_absdif

    anomaly_points = np.where(absolute_difference >= threshold)
    anomaly_points += test_set['Volume'].index[0]

    with open(f'./src/pca/pickles/{VERSION}_anomalous_points_PCA({i})_gamma_{GAMMA}.pkl', 'wb') as pickle_file:
        pickle.dump(anomaly_points, pickle_file)

    plt.figure(figsize=(19.20, 10.80))
    plt.title(f'Diferenta in modul intre y si y_rec (PCA({i}))')
    plt.plot(test_set['Volume'].keys() - test_set['Volume'].index[0], absolute_difference, label = 'Diferenta in modul')
    plt.hlines(mean_absdif, 0, absolute_difference.shape[0], label = 'Media diferentelor', linestyles='solid', color = 'blue')
    plt.hlines(threshold, 0, absolute_difference.shape[0], label = f'Threshold (medie + {NUM_STD_DEV}*std)', linestyles='dashed', color = 'black')
    plt.savefig(f"./src/pca/plots/{VERSION}_diferenta_modul_PCA({i})_gamma_{GAMMA}.svg", format='svg')
    plt.legend()

    plt.figure(figsize=(19.20, 10.80))
    plt.title(f'Anomalii detectate (PCA({i}))')
    plt.plot(test_set['Volume'], label = 'Volumul observat intr-o zi')
    plt.plot(test_set['Volume'].loc[anomaly_points[0]], 'ro', label = 'Punct de anomalie')
    plt.legend()
    plt.savefig(f"./src/pca/plots/{VERSION}_anomalii_detectate_PCA({i})_gamma_{GAMMA}.svg", format='svg')

plt.show()