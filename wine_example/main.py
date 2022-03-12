# Getting started with keras by:
#   https://www.datacamp.com/community/tutorials/deep-learning-python

import pandas as pd
import matplotlib.pyplot as plt

white = pd.read_csv('wine_example/data/winequality-white.csv', sep=';')
red = pd.read_csv('wine_example/data/winequality-red.csv', sep=';')

white['type'] = 'white'
red['type'] = 'red'

all_data = pd.concat([white, red])

# Data Exploration
white.info()
red.info()
white.describe()
red.head()
pd.isnull(red)
red.isnull().sum()
white.isnull().sum()
red.sample(5)

# Data Visualization

fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5,
           label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
fig.suptitle("Distribution of Alcohol in % Vol")

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
