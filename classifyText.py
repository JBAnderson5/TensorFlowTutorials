#following tutorial: https://www.tensorflow.org/alpha/tutorials/keras/basic_text_classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


imdb = keras.datasets.imdb

(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words = 1000)

print("training entries: {}, labels: {}".format(len(trainData), len(trainLabels)))

print("data point",trainData[0])

print("length of two data points: {} {}".format(len(trainData[0]),len(trainData[1])))

wordIndex = imdb.get_word_index()

wordIndex = {k:(v+3) for k,v in wordIndex.items()}
wordIndex["<PAD>"] = 0
wordIndex["<START>"] = 1
wordIndex["<UNK>"] = 2 #unkown
wordIndex["<UNUSED>"] = 3

reverseWordIndex = dict([ (value,key) for (key,value) in wordIndex.items()])

def decodeReview(text):
    return " ".join([reverseWordIndex.get(i, "?") for i in text])

print("first review", decodeReview(trainData[0]))


trainData = keras.preprocessing.sequence.pad_sequences(trainData, value = wordIndex["<PAD>"],
    padding = "post", maxlen = 256)
testData = keras.preprocessing.sequence.pad_sequences(testData, value = wordIndex["<PAD>"],
    padding = "post", maxlen = 256)

print("new data point lengths: {} {}".format(len(trainData[0]), len(trainData[1])))
print("new first review", trainData[0])

vocabSize = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocabSize, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

xVal = trainData[:10000]
partialXTrain = trainData[10000:]

yVal = trainLabels[:10000]
partialYTrain = trainLabels[10000:]

history = model.fit(partialXTrain,partialYTrain, epochs = 40, batch_size = 512,
    validation_data = (xVal,yVal), verbose = 1)

results = model.evaluate(testData,testLabels)
print(results)

historyDict = history.history
print("keys",historyDict.keys())


acc = historyDict["accuracy"]
valAcc = historyDict["val_accuracy"]
loss = historyDict["loss"]
valLoss = historyDict["val_loss"]

epochs = range(1,len(acc) + 1)

plt.plot(epochs,loss,"bo",label = "Training Loss")
plt.plot(epochs,valLoss,"b", label = "Validation Loss")
plt.title("Training and validation Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs,acc,"bo", label = "Training Accurcacy")
plt.plot(epochs,valAcc,"b", label = "Validation Accurcacy")
plt.title("Training and Validation Accurcay")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
