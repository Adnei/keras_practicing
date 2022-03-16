import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop

white = pd.read_csv('wine_example/data/winequality-white.csv', sep=';')
red = pd.read_csv('wine_example/data/winequality-red.csv', sep=';')

white['type'] = 0
red['type'] = 1

wines = red.append(white, ignore_index=True)

###############################################################################
#      In this case, the quality prediction is a single continuous value      #
###############################################################################

y = wines.quality
X = wines.drop('quality', axis=1)
X = StandardScaler().fit_transform(X)
# ____________________________________________________________________________
#                                                                             #
#                                 KFold Validation                            #
# ____________________________________________________________________________#
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, y):
    model = Sequential()
    # input layer
    model.add(Dense(128, input_dim=12, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # model.add(Dense(6, activation='relu'))
    # For scalar regression -> It is common to the output layer to be a single
    # unit with no activation fn
    model.add(Dense(1))
    # sgd = SGD(learning_rate=0.1)
    # model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    rmsprop = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])
    # model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(X[train], y[train], epochs=10, verbose=1)

# _____________________________________________________________________________

y_pred = model.predict(X[test])
mse_value, mae_value = model.evaluate(X[test], y[test], verbose=0)

print(mse_value, 'reference -> 0')
print(mae_value, 'reference -> 1')
print(r2_score(y[test], y_pred), 'reference -> 1')
