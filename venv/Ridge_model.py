import csv
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
from keras import models
from keras import layers
from sklearn import metrics

NUM_EPOCHS = 100
BATCH_SIZE = 512
train_data = pd.read_csv('train_data.txt', sep=' ', header=None)
train_targets = pd.read_csv('train_targets.txt', sep=' ', header=None)
test_data = pd.read_csv('test_data.txt', sep=' ', header=None)
test_targets = pd.read_csv('test_targets.txt', sep=' ', header=None)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model = build_model()
model.fit(train_data, train_targets, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
results = model.evaluate(test_data, test_targets)
print(results)
