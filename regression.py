#following tutorial: https://www.tensorflow.org/alpha/tutorials/keras/basic_regression

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataSetPath = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataSetPath

columnNames = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration",
    "Model Year", "Origin"]

rawDataset = pd.read_csv(dataSetPath, names = columnNames, na_values = "?",
    comment = "\t", sep = " ", skipinitialspace = True)

dataset = rawDataset.copy()
print(dataset.tail())

dataset.isna().sum()

dataset = dataset.dropna()

origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0
print(dataset.tail())

trainDataSet = dataset.sample(frac = 0.8, random_state = 0)
testDataSet = dataset.drop(trainDataSet.index)

sns.pairplot(trainDataSet[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind = "kde")
plt.show()
