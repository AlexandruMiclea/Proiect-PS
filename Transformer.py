import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import time
def Positional_Encoding(dim_subsecventa, dim_embedding):
    factori_scalare = np.array([1 / (10000 ** (2 * (pozitie_embedding // 2) / dim_embedding)) for pozitie_embedding in range(dim_embedding)])  # (1, dim_embedding)
    pozitii_initiale = np.array([[p] for p in range(dim_subsecventa)])  # (dim_subsecventa, 1)

    valori = pozitii_initiale * factori_scalare  # (dim_subsecventa, dim_embedding)
    # esantioanele au initial pozitiile 0, 1, 2, etc,
    # pozitiile vor deveni arrays de dimensiune dim_embedding,
    # fiecare element din embedding fiind pozitia initiala a esantionului * factor de scalare
    rezultat = np.zeros((dim_subsecventa, dim_embedding))
    rezultat[:, 0::2] = np.sin(valori[:, 0::2])
    rezultat[:, 1::2] = np.cos(valori[:, 1::2])
    return rezultat

def Self_Attention(layer_precedent, dim_embedding):
    # Q = ce informatie cauta un esantion de la altele, K = ce informatie detine fiecare, V = informatie deitnuta in detaliu
    Q = layers.Dense(dim_embedding)(layer_precedent)
    # transofrmam datele din vectorul de embedding pentru a aprofunda informatia deja existenta
    K = layers.Dense(dim_embedding)(layer_precedent)
    V = layers.Dense(dim_embedding)(layer_precedent)

    scoruri_atentie = layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([Q, K])
    # scoruri_atentie e de dimensiune (lungime_secventa, lungime_secventa)
    # deci fiecare esantion din secventa are un scor de atentie fata de restul
    # prin Q * K.T fiecare esantion vede daca are ce obtine de la restul
    ponderi_atentie = layers.Softmax(axis=-1)(scoruri_atentie)
    # fiecare esantion primeste de la fiecare ce a cautat
    rezultat = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([ponderi_atentie, V])
    return rezultat

def Encoder(layer_precedent, dim_embedding, dim_feed_forward):
    self_attention = Self_Attention(layer_precedent, dim_embedding)
    self_attention += layer_precedent
    self_attention = layers.LayerNormalization()(self_attention)

    feed_forward = layers.Dense(dim_feed_forward, activation='relu')(self_attention)
    # primul strat din ff mareste dimensiunea pentru a aprofunda informatia din self_attention, iar al doilea aduce dimensiunea la loc pentru a se potrivi cu dim_encoder
    feed_forward = layers.Dense(dim_embedding)(feed_forward)

    encoder = feed_forward + self_attention
    encoder = layers.LayerNormalization()(encoder)
    return encoder

def Transformer(dim_subsecventa, dim_embedding, nr_encoders, dim_feed_forward):
    tensor_intrare = Input(shape=(dim_subsecventa, dim_embedding))
    pos_encoding = Positional_Encoding(dim_subsecventa, dim_embedding)

    tensor_pos_encoding = tf.convert_to_tensor(pos_encoding, dtype=tf.float32)
    # adaugam inca o dimensiune la tensor pt a sti din ce batch face parte
    tensor_pos_encoding = tf.expand_dims(tensor_pos_encoding, axis=0)
    layer_pos_encoding = tensor_intrare + tensor_pos_encoding

    layers_encoder = layer_pos_encoding
    for _ in range(nr_encoders):
        layers_encoder = Encoder(layers_encoder, dim_embedding, dim_feed_forward)
    # layers encoder are acum dim (batch, dim_subsecventa, dim_embedding)

    # obtinem informatie despre fiecare subsecventa
    layer_medie_pe_subsecvente = layers.GlobalAveragePooling1D()(layers_encoder)
    # pt a returna o singura val, adica predictia urm, trb sa folosim toate informatiile acumulate
    layer_final = layers.Dense(1)(layer_medie_pe_subsecvente)
    # layer final cu un singur neuron la iesire, 1 * dim = 1 valoare
    return Model(tensor_intrare, layer_final)


dim_subsecventa = 30
nr_encoders = 2
dim_embedding = 64
dim_feed_forward = 256

df = pd.read_csv("datasets/NVidia_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df_filtrat = df[(df['Date'].dt.year >= 2022) & (df['Date'].dt.year <= 2024)]
valori = df_filtrat[["Open", "High", "Low", "Close"]].values
low_values = df_filtrat["High"].values
serie_timp = np.array(low_values)
N = len(serie_timp)
X = np.array([serie_timp[i:i + dim_subsecventa] for i in range(N - dim_subsecventa)])
y = serie_timp[dim_subsecventa:]

X = np.expand_dims(X, axis=-1)
X = np.repeat(X, dim_embedding, axis=-1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Transformer(dim_subsecventa, dim_embedding, nr_encoders, dim_feed_forward)
model.compile(optimizer='adam', loss='mse')

print("Incepe antrenarea")
start_time = time.time()
model.fit(X_train, y_train, epochs=18, batch_size=32, validation_data=(X_test, y_test))
end_time = time.time()
difference_time = end_time - start_time
print(f"Antrenarea a durat: {difference_time:.4f} secunde.")

start_time = time.time()
predictions = model.predict(X_test)
end_time = time.time()
difference_time = end_time - start_time
print(f"Testarea a durat: {difference_time:.4f} secunde.")

erori = y_test - predictions.flatten()
media = np.mean(erori)
dev_standard = np.std(erori)

z_scores = (erori - media) / dev_standard

prag = 2
indici_anomalii = np.where((z_scores > prag) | (z_scores < -prag))[0]

# plt.figure(figsize=(14, 7))
# plt.plot(y_test, label='Semnal Real', color='blue')
# plt.plot(predictions, label='Predicții', color='red', linestyle='--')
# plt.scatter(indici_anomalii, y_test[indici_anomalii], color='yellow', s=50, label='Anomalii', marker='o')
# plt.legend()
# plt.savefig("Transformer_images/detectie_anomalii.pdf", format="pdf")
# plt.show()

# #
# # # Detectie anomalii pe serie cu componente aleatoare
# N = 1000
# t = np.linspace(0, 1, N)
# trend = 10 * t * t + 0.1 * t + 0.2
# seasonal = 3 * np.sin(2 * np.pi * t * 10) + 2 * np.sin(2 * np.pi * t * 80)
# noise = np.random.normal(0, 1.8, N)
# serie_timp = trend + seasonal + noise
#
# N = len(serie_timp)
# X = np.array([serie_timp[i:i + dim_subsecventa] for i in range(N - dim_subsecventa)])
# y = serie_timp[dim_subsecventa:]
#
# X = np.expand_dims(X, axis=-1)
# X = np.repeat(X, dim_embedding, axis=-1)
# y = np.array(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#
# model = Transformer(dim_subsecventa, dim_embedding, nr_encoders, dim_feed_forward)
# model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
# predictions = model.predict(X_test)
#
#
# erori = y_test - predictions.flatten()
# media = np.mean(erori)
# dev_standard = np.std(erori)
#
# z_scores = (erori - media) / dev_standard
#
# prag = 2.5
# indici_anomalii = np.where((z_scores > prag) | (z_scores < -prag))[0]
#
# plt.figure(figsize=(14, 7))
# plt.plot(y_test, label='Semnal Real', color='blue')
# plt.plot(predictions, label='Predicții', color='red', linestyle='--')
# plt.scatter(indici_anomalii, y_test[indici_anomalii], color='green', s=50, label='Anomalii', marker='o')
# plt.legend()
# plt.savefig("Transformer_images/detectie_anomalii_serie_cu_putine_elemente_random.pdf", format="pdf")
# plt.show()