import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def IsolationForest(X, nr_arbori = 300, dim_subsecventa = 216, prag_anomalie = 0.80):
    N = len(X)
    dim_subsecventa = min(dim_subsecventa, N // 2)
    adancime_maxima = np.log2(dim_subsecventa)
    subsecvente = []

    for i in range(nr_arbori):
        indici_alesi = np.random.choice(N, size=dim_subsecventa, replace=False)
        subsecvente.append(X[indici_alesi])

    padure = []
    for subsecventa in subsecvente:
        padure.append((subsecventa, CreareArbore(subsecventa, 0, adancime_maxima)))

    H_N = np.log(N - 1) + 0.577215
    constanta_normalizare = 2 * H_N - (2 * (N - 1) / N)
    scoruri_anomalie = []
    rezultat = []
    for x in X:
        adancimi = []
        for subsecventa, arbore in padure:
            if np.any(np.all(subsecventa == x, axis=1)):
                adancime = AdancimeInArbore(arbore, x)
                adancimi.append(adancime)

        adancime_medie = np.mean(adancimi)
        scor_anomalie = 2 ** (-(adancime_medie / constanta_normalizare))
        scoruri_anomalie.append(scor_anomalie)
        if scor_anomalie > prag_anomalie:
            rezultat.append(1)
        else:
            rezultat.append(0)
        print(adancime_medie, constanta_normalizare, 2 ** (-(adancime_medie / constanta_normalizare)))

    #return scoruri_anomalie
    return rezultat

def AdancimeInArbore(arbore, x):
    if len(arbore) == 1:
        return 0

    trasatura = arbore[0]
    valoare_trasatura = arbore[1]
    if x[trasatura] <= valoare_trasatura:
        return 1 + AdancimeInArbore(arbore[2], x)
    else:
        return 1 + AdancimeInArbore(arbore[3], x)

def CreareArbore(subsecventa, adancime_curenta, adancime_maxima):
    nr_esantioane, nr_trasaturi = subsecventa.shape
    if adancime_curenta >= adancime_maxima or nr_esantioane <= 1:
        return [nr_esantioane]

    if len(subsecventa) == 0:
        return []

    trasatura = np.random.choice(nr_trasaturi)
    valoare_trasatura = np.random.uniform(np.min(subsecventa[:, trasatura]), np.max(subsecventa[:, trasatura]))

    esantioane_stanga = subsecventa[subsecventa[:, trasatura] <= valoare_trasatura]
    esantioane_dreapta = subsecventa[subsecventa[:, trasatura] > valoare_trasatura]

    if len(esantioane_stanga) == 0:
        subarbore_stang = None
    else:
        subarbore_stang = CreareArbore(esantioane_stanga, adancime_curenta + 1, adancime_maxima)
    if len(esantioane_dreapta) == 0:
        subarbore_drept = None
    else:
        subarbore_drept = CreareArbore(esantioane_dreapta, adancime_curenta + 1, adancime_maxima)

    return [trasatura, valoare_trasatura, subarbore_stang, subarbore_drept]


# anii 2022 2023 2024, afisam doar 2024
# df = pd.read_csv("datasets/NVidia_stock_history.csv")
# df['Date'] = pd.to_datetime(df['Date'], utc=True)
# df_filtrat = df[(df['Date'].dt.year >= 2022) & (df['Date'].dt.year <= 2024)]
# nr_esantioane_2024 = len(df_filtrat[df_filtrat['Date'].dt.year == 2024])
# valori = df_filtrat["Volume"].values.reshape(-1, 1)
# rezultat = IsolationForest(valori)
#
#
# plt.figure(figsize=(14, 7))
# valori = valori[-nr_esantioane_2024:]
# rezultat = rezultat[-nr_esantioane_2024:]
#
# indici_anomalii = []
# for i in range(len(valori)):
#     if rezultat[i] == 1:
#         indici_anomalii.append(i)
# plt.plot(valori, label="serie timp")
# plt.scatter(indici_anomalii, valori[indici_anomalii], color="red", s=10, label="anomalii")
# plt.legend()
# plt.savefig("IsolationForest_images/trasatura_Volume_2022-2024.pdf", format="pdf")
# plt.show()
#
# with open("anomalii.pkl", "wb") as f:
#     pickle.dump(indici_anomalii, f)

#TOT DATASETUL TESTAT PE  TRASATURA "Close"
df = pd.read_csv("datasets/NVidia_stock_history.csv")
valori = df["Close"].values.reshape(-1, 1)

rezultat = IsolationForest(valori)

plt.figure(figsize=(14, 7))
indici_anomalii = []
for i in range(len(valori)):
    if rezultat[i] == 1:
        indici_anomalii.append(i)
plt.plot(valori, label="serie timp")
plt.scatter(indici_anomalii, valori[indici_anomalii], color="red", s=10, label="anomalii")
plt.legend()
plt.savefig("IsolationForest_images/trasatura_Close.pdf", format="pdf")
plt.show()



# TOT DATASETUL TESTAT CU MULTIPLE CARACTERISTICI
# date1 = pd.read_csv("datasets/NVidia_stock_history.csv")
# date = date1[5700:5900]
# valori = date[["Open", "High", "Low", "Close"]].values
#
# rezultat_IF = IsolationForest(valori)
#
# # interval_stanga = 5000
# # interval_dreapta = 6000
#
# plt.figure(figsize=(14, 7))
#
# plt.plot(date["Close"], label="Close", color="blue")
# plt.plot(date["Open"], label="Open", color="green")
# plt.plot(date["High"], label="High", color="orange")
# plt.plot(date["Low"], label="Low", color="purple")
#
# for i in range(len(rezultat_IF)):
#     if rezultat_IF[i] == 1:
#         plt.axvline(x= i + 5700, color='red', linestyle='--', linewidth=1)
#
# plt.legend()
# plt.savefig("IsolationForest_images/toate_trasaturile.pdf", format="pdf")
# plt.show()


# X = np.array([1.2, 2.5, 3.1, 4.7, 5.0, 100.0, 2.3, 1.9])
# X = X.reshape(-1, 1)
#
# anomalii = IsolationForest(X)
#
# print(anomalii)
#
# timp_normal = np.linspace(0, 1000, 1000)
# valori_normal = np.sin(timp_normal) + np.random.normal(0, 0.1, 1000)
#
# timp_anomal = np.array([30, 50, 75])
# valori_anomal = np.array([5, -5, 10])
#
# valori_normal[timp_anomal.astype(int)] = valori_anomal
# valori_normal = valori_normal.reshape(-1, 1)
#
# scoruri = IsolationForest(valori_normal)
# print(scoruri[29], scoruri[50], scoruri[75])
