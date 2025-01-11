import os
import numpy as np
import torch
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
from keras import backend as K
from matplotlib import pyplot as plt
from show_dataset import *

def show_dataset_info(dataset):
    _, ax1 = plt.subplots(2,1)
    _, ax2 = plt.subplots(2,1)
    _, ax3 = plt.subplots(3,1)
    dataset["Open"].plot(legend=True, ax = ax1[0])
    dataset["Close"].plot(legend=True, ax = ax1[1])
    dataset["High"].plot(legend=True, ax = ax2[0])
    dataset["Low"].plot(legend=True, ax = ax2[1])
    dataset["Volume"].plot(legend=True, ax = ax3[0])
    dataset["Dividends"].plot(legend=True, ax = ax3[1])
    dataset["Stock Splits"].plot(legend=True, ax = ax3[2])
    plt.show()