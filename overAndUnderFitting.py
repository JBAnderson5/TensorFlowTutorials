#following tutorial: https://www.tensorflow.org/alpha/tutorials/keras/overfit_and_underfit

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

NUM_WORDS = 10000

(trainData, trainLabels), (testData,testLabels) = keras.datasets.imdb.load_data(num_words = NUM_WORDS)

def multiHotSequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, wordIndices in enumerate(sequences):
        results[i,wordIndices] = 1.0
    return results

trainData = multiHotSequences(trainData,NUM_WORDS)
testData = multiHotSequences(testData,NUM_WORDS)

plt.plot(trainData[0])
plt.show()

baseLineModel = keras.Sequential([
    keras.layers.Dense(16, activation = "relu", input_shape = (NUM_WORDS,)),
    keras.layers.Dense(16, activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])

baseLineModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
baseLineModel.summary()

baseLineHistory = baseLineModel.fit(trainData, trainLabels, epochs = 20,
    batch_size = 512, validation_data = (testData, testLabels),verbose = 2)

smallerModel = keras.Sequential([
    keras.layers.Dense(4, activation = "relu", input_shape = (NUM_WORDS,)),
    keras.layers.Dense(4, activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])

smallerModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
smallerModel.summary()

smallerHistory = smallerModel.fit(trainData, trainLabels, epochs = 20,
    batch_size = 512, validation_data = (testData,testLabels), verbose = 2)

biggerModel = keras.models.Sequential([
    keras.layers.Dense(512, activation = "relu", input_shape = (NUM_WORDS,)),
    keras.layers.Dense(512, activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])

biggerModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
biggerModel.summary()

biggerHistory = biggerModel.fit(trainData, trainLabels, epochs = 20,
    batch_size = 512, validation_data = (testData,testLabels), verbose = 2)

def plotHistory(histories, key = "binary_crossentropy"):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history["val_"+key], "--", label = name.title() + " Val")
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(), label = name.title() + " train")
        plt.xlabel("epochs")
        plt.ylabel(key.replace("_"," ").title())
        plt.legend()
        plt.xlim([0,max(history.epoch)])
    plt.show()


plotHistory([("baseline", baseLineHistory),
            ("smaller", smallerHistory),
            ("bigger", biggerHistory)])

l2Model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001), activation = "relu", input_shape = (NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001), activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])
l2Model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
l2Model.summary()
l2History = l2Model.fit(trainData, trainLabels, epochs = 20, batch_size = 512,
    validation_data = (testData,testLabels), verbose = 2)

plotHistory([("baseline", baseLineHistory),
            ("l2",l2History)])

dropoutModel = keras.models.Sequential([
    keras.layers.Dense(16, activation = "relu", input_shape = (NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = "sigmoid")
])
dropoutModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "binary_crossentropy"])
dropoutModel.summary()
dropoutHistory = dropoutModel.fit(trainData, trainLabels, epochs = 20, batch_size = 512,
    validation_data = (testData, testLabels), verbose = 2)

plotHistory([("baseline", baseLineHistory),
            ("dropout",dropoutHistory)])
