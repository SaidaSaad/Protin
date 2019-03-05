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

train_targets = pd.read_csv('train_targets.txt', sep=' ')
train_targets['Unnamed: 0'] = train_targets['Unnamed: 0'].astype('int')
train_targets = train_targets.set_index('Unnamed: 0')

test_data = pd.read_csv('test_data.txt', sep=' ')
test_data['Unnamed: 0'] = test_data['Unnamed: 0'].astype('int')
test_data = test_data.set_index('Unnamed: 0')

test_targets = pd.read_csv('test_targets.txt', sep=' ', header = None)
test_targets = test_targets.drop(0, axis=0)
test_targets[0] = test_targets[0].astype('int')
test_targets = test_targets.set_index(0)

train_data = pd.read_csv('train_data.txt', sep=' ', header = None)
train_data = train_data.drop(0, axis=0)
train_data[0] = train_data[0].astype('int')
train_data = train_data.set_index(0)

print(train_data)
print(test_data)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


model = build_model()
model.fit(train_data, train_targets, epochs=2, batch_size=512, verbose=1)
results = model.evaluate(test_data, test_targets)
print(results)
