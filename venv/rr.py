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
from keras.callbacks import EarlyStopping
from keras import optimizers
NUM_EPOCHS = 100
BATCH_SIZE = 512


data = pd.read_csv('big_table_0.2.txt', sep=' ', header=None)
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
print(data.shape)

X = data.loc[:, :2004]
y = data.loc[:,  2005]

U = X.columns[(X == 0).all()]
print(U)
X = X.loc[:, (X != 0).any(axis=0)]

train_data, test_data, train_targets, test_targets = train_test_split(X, y, test_size=0.3, random_state=42)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = train_data-mean
train_data /= std
test_data -= mean
test_data /= std

train_data = train_data.drop(train_data.columns[153], axis=1)
test_data = test_data.drop(test_data.columns[153], axis=1)

r = np.where(np.asanyarray(np.isnan(train_data)))
r1 = np.where(np.asanyarray(np.isnan(test_data)))
print(r)
print(r1)

print(y.min(), y.max(), y.mean())
#
# train_data.to_csv('train_data.txt', header='False', index='False', sep=' ')
# train_targets.to_csv('train_targest.txt', header='False', index='False', sep=' ')
# test_data.to_csv('test_data.txt', header='False', index='False', sep=" ")
# test_targets.to_csv('test_targets.txt', header='False', index='False', sep=' ')



# lr = LinearRegression().fit(train_data, train_targets)
# print("lr.Coef_:{}".format(lr.coef_))
# print("lr.intercept_:{}".format(lr.intercept_))
# test_pred = lr.predict(test_data)
# #df = pd.DataFrame({'Actual': test_targets, 'Predicted': y_pred})
# print('Mean Absolute Error:', metrics.mean_absolute_error(test_targets, test_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(test_targets, test_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_targets, test_pred)))
# #print("The confidence intervals for the model coefficients", lr.conf_int())
# #print("P-values for the model coefficients", lr.pvalues)
# #print("Training Score:{:.2f}".format(lr.score(train_data, train_targets)))
# #print("Test Score:{:.2f}".format(lr.score(test_data, test_targets)))
# #
# ridge1 = Ridge().fit(train_data, train_targets)
# # print("lr.Coef_:{}".format(ridge1.coef_))
# # print("lr.intercept_:{}".format(ridge1.intercept_))
# test_pred = ridge1.predict(test_data)
# print('Mean Absolute Error:', metrics.mean_absolute_error(test_targets, test_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(test_targets, test_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_targets, test_pred)))
# # print("Training Score:{:.2f}".format(ridge1.score(train_data, train_targets)))
# # print("Test Score:{:.2f}".format(ridge1.score(test_data, test_targets)))
# #
# ridge01 = Ridge(alpha=0.1).fit(train_data, train_targets)
# # print("lr.Coef_0.1:{}".format(ridge01.coef_))
# # print("lr.intercept_0.1:{}".format(ridge01.intercept_))
# test_pred = ridge01.predict(test_data)
# print('Mean Absolute Error:', metrics.mean_absolute_error(test_targets, test_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(test_targets, test_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_targets, test_pred)))
# # print("Training Score:{:.2f}".format(ridge01.score(train_data, train_targets)))
# # print("Test Score:{:.2f}".format(ridge01.score(test_data, test_targets)))
# #
# lasso1 = Lasso(alpha=0.001, max_iter=10000).fit(train_data, train_targets)
# # print("lasso1.Coef_:{}".format(lasso1.coef_))
# # print("lasso1.intercept_:{}".format(lasso1.intercept_))
# test_pred = lasso1.predict(test_data)
# print('Mean Absolute Error:', metrics.mean_absolute_error(test_targets, test_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(test_targets, test_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_targets, test_pred)))
# # print("Training Score:{:.2f}".format(lasso1.score(train_data, train_targets)))
# # print('Test Score:{:.2f}'.format(lasso1.score(test_data, test_targets)))

test_targets.to_csv('results.csv', header='False', index='False', sep=' ')

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(1))
    sgd = optimizers.SGD(lr=0.1, clipnorm=1.)
    model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    return model
print(train_data)
print(test_data)
model = build_model()
model.fit(train_data, train_targets, epochs=500, batch_size=512, verbose=1, callbacks=callbacks)
ynew = model.predict(test_data)
# show the inputs and predicted outputs
#for i in range(len(ynew)):
#    print("Predicted=%s" % (ynew[i]))
results = model.evaluate(test_data, test_targets)
print(results)

print(test_targets)
x = test_targets.values
print(x)

test_targets.to_csv('results.csv', header= ['Targets'], sep=' ')

#### adding Predictions results
df = pd.read_csv('results.csv', sep = ' ')
df['Prediction'] = ynew
df.to_csv('results.csv', sep = ' ', index = False)
print(df)


# df = pd.read_csv('results.csv')
# df['Targets'] = test_targets.values
# df['Prediction'] = ynew
# df.to_csv('results.txt', sep=' ', )
