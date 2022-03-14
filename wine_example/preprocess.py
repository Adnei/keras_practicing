import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

white = pd.read_csv('wine_example/data/winequality-white.csv', sep=';')
red = pd.read_csv('wine_example/data/winequality-red.csv', sep=';')

white['type'] = 0
red['type'] = 1

wines = red.append(white, ignore_index=True)

corr = wines.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# ignoring quality and type
X = wines.iloc[:, 0:11]

# imbalanced data --> needs to be flatten
y = np.ravel(wines.type)
# y = wines.type

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Ok, my data is way too scattered. Therefore, I need to standardize it
# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)
