###############################################################################
# @TODO --> Experiments                                                       #
#        -> Try it out with more or less layers                               #
#        -> Make a multi-class classification problem by considering the      #
#     'quality' property                                                      #
#        -> Try out changing the activation function                          #
#           -> learn more about activation functions                          #
#        -> Try to predict the wine quality                                   #
###############################################################################
import wine_example.preprocess as pre
from keras.models import Sequential
from keras.layers import Dense
# separing imports just to break the line
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import f1_score, cohen_kappa_score

model = Sequential()

# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='tanh'))
model.add(Dense(6, activation='tanh'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))

# _____________________________________________________________________________

model.summary()
model.get_config()
model.get_weights()

# _____________________________________________________________________________

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(pre.X_train, pre.y_train, epochs=10, batch_size=1, verbose=1)
y_pred = model.predict(pre.X_test)
y_pred_bin = list(map(lambda x: 0 if x < 0.5 else 1, y_pred))

score = model.evaluate(pre.X_test, pre.y_test, verbose=1)

confusion_matrix(pre.y_test, y_pred_bin)

precision_score(pre.y_test, y_pred_bin)
f1_score(pre.y_test, y_pred_bin)
cohen_kappa_score(pre.y_test, y_pred_bin)
