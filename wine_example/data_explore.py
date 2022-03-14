# Getting started with keras by:
#   https://www.datacamp.com/community/tutorials/deep-learning-python
# pandas basics
#   https://www.w3resource.com/python-exercises/pandas/index.php
#   https://www.machinelearningplus.com/python/101-pandas-exercises-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

white = pd.read_csv('wine_example/data/winequality-white.csv', sep=';')
red = pd.read_csv('wine_example/data/winequality-red.csv', sep=';')

# Data Exploration
white.info()
red.info()
white.describe()
red.head()
pd.isnull(red)
red.isnull().sum()
white.isnull().sum()
red.sample(5)

# _____________________________________________________________________________
# _____________________________________________________________________________

# Data Visualization

fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label='Red wine')
ax[1].hist(white.alcohol, 10, facecolor='white', ec='black', lw=0.5, alpha=0.5,
           label='White wine')

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel('Alcohol in % Vol')
ax[0].set_ylabel('Frequency')
ax[1].set_xlabel('Alcohol in % Vol')
ax[1].set_ylabel('Frequency')
fig.suptitle('Distribution of Alcohol in % Vol')

plt.show()

# Quality Comparison
# @TODO: Make a graphical comparison: red quality vs. white quality
white_sample = white.sample(len(red.index))
plt.hist(white_sample.quality)
plt.hist(red.quality)

# bucketing or binning
# white['binning'] = pd.qcut(
#     white['quality'], q=10, duplicates='drop')
# red['binning'] = pd.qcut(red['quality'], q=100, duplicates='drop')

# _____________________________________________________________________________
# _____________________________________________________________________________

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red['quality'], red['sulphates'], color='red')
ax[1].scatter(white['quality'], white['sulphates'],
              color='white', edgecolors='black', lw=0.5)

ax[0].set_title('Red Wine')
ax[1].set_title('White Wine')
ax[0].set_xlabel('Quality')
ax[1].set_xlabel('Quality')
ax[0].set_ylabel('Sulphates')
ax[1].set_ylabel('Sulphates')
ax[0].set_xlim([0, 10])
ax[1].set_xlim([0, 10])
ax[0].set_ylim([0, 2.5])
ax[1].set_ylim([0, 2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle('Wine Quality by Amount of Sulphates')

plt.show()

# _____________________________________________________________________________
# _____________________________________________________________________________

np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6, 4)
whitecolors = np.append(redcolors, np.random.rand(1, 4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0, 1.7])
ax[1].set_xlim([0, 1.7])
ax[0].set_ylim([5, 15.5])
ax[1].set_ylim([5, 15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol")
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
fig.subplots_adjust(top=0.85, wspace=0.7)

plt.show()
