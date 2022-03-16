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
model.fit(pre.X_train, pre.y_train, epochs=20, batch_size=1, verbose=1)

y_pred_prob = model.predict(pre.X_test)
# Turns out the model returns the probability of the class to be 0 or 1
# In this case, when the probability is > 0.5, then it is likely to be class 1
# If the probability is < 0.5, then it is likely to be class 0
# If the probability is around 0.5, this means the model is somewhat "unsure"
# Ideal scenario --> all the probability as close as possible to 0 or 1
y_pred = list(map(lambda x: 0 if x < 0.5 else 1, y_pred_prob))

# evaluating the model

score = model.evaluate(pre.X_test, pre.y_test, verbose=1)

# This is a good way to analyze false positives, false negatives, etc...
confusion_matrix(pre.y_test, y_pred)

precision_score(pre.y_test, y_pred)
f1_score(pre.y_test, y_pred)
cohen_kappa_score(pre.y_test, y_pred)
