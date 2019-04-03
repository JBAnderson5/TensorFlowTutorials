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

trainStats = trainDataSet.describe()
trainStats.pop("MPG")
trainStats = trainStats.transpose()
print(trainStats)

trainLabels = trainDataSet.pop("MPG")
testLabels = testDataSet.pop("MPG")

def norm(X):
    return (X -trainStats["mean"]) / trainStats["std"]

normTrainData = norm(trainDataSet)
normTestData = norm(testDataSet)

def buildModel():
    model = keras.Sequential([
        layers.Dense(64, activation = "relu", input_shape = [len(trainDataSet.keys())]),
        layers.Dense(64, activation = "relu"),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse",optimizer=optimizer,metrics = ["mae","mse"])

    return model

model = buildModel()

model.summary()

exampleBatch = normTrainData[:10]
exampleResult = model.predict(exampleBatch)
print("example result",exampleResult)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0: print(" ")
        print(".",end="")

EPOCHS = 10000

history = model.fit(normTrainData, trainLabels, epochs = EPOCHS,
    validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print(hist.tail())

def plotHistory(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("EPOCH")
    plt.ylabel("mean abs error [MPG]")
    plt.plot(hist["epoch"],hist["mae"],label = "Train error")
    plt.plot(hist["epoch"],hist["val_mae"],label = "Val error")
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel("EPOCH")
    plt.ylabel("mean square error [$MPG^2$]")
    plt.plot(hist["epoch"],hist["mse"],label = "Train error")
    plt.plot(hist["epoch"],hist["val_mse"],label = "val error")
    plt.ylim([0,20])
    plt.legend()
    plt.show()

plotHistory(history)

model = buildModel()
#patience is the number of epochs between checking for improvement
earlyStop = keras.callbacks.EarlyStopping(monitor = "val_loss",patience = 10)

history = model.fit(normTrainData, trainLabels, epochs = EPOCHS,
    validation_split = 0.2, verbose = 0, callbacks = [earlyStop, PrintDot()])

plotHistory(history)

loss, mae, mse = model.evaluate(normTestData,testLabels, verbose=0)
print("Testing set mean absolute error {:5.2f} MPG".format(mae))

testPredictions = model.predict(normTestData).flatten()

plt.scatter(testLabels,testPredictions)
plt.xlabel("true values [mpg]")
plt.ylabel("predicted values [mpg]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100,100],[-100,100])
plt.show()

error = testPredictions - testLabels
plt.hist(error,bins = 25)
plt.xlabel("prediction errors [MPG]")
plt.ylabel("Count")
plt.show()
