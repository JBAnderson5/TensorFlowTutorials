#following tutorial: https://www.tensorflow.org/alpha/tutorials/keras/basic_classification

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashionMnist = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages,testLabels) = fashionMnist.load_data()

classNames = ["tshirt/top", "trouser", "pullover", "dress",
    "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

print("training image shape", trainImages.shape)
print("length of training labels", len(trainLabels))
print("training labels", trainLabels)

print("test image shape", testImages.shape)
print("length of test labels", len(testLabels))

plt.figure()
plt.imshow(trainImages[0])
plt.colorbar()
plt.grid(False)
plt.show()

trainImages = trainImages / 255.0
testImages = testImages / 255.0

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i],cmap=plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax")
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(trainImages,trainLabels, epochs=5)

testLoss,testAccuracy = model.evaluate(testImages, testLabels)
print("test accuracy", testAccuracy)

predictions = model.predict(testImages)
print("image 0")
print("prediction array", predictions[0])
print("class chosen", np.argmax(predictions[0]))
print("actual class label", testLabels[0])

def plotImages(i,predictionsArray, trueLabel, img):
    predictionsArray, trueLabel, img = predictionsArray[i], trueLabel[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predictionLabel = np.argmax(predictionsArray)
    if predictionLabel == trueLabel:
        color = "blue"
    else:
        color = "red"

    plt.xlabel("{} {:0.2f}% ({})".format(classNames[predictionLabel],
        100*np.max(predictionsArray), classNames[trueLabel]), color = color)

def plotValueArray(i, predictionsArray, trueLabel):
    predictionsArray, trueLabel = predictionsArray[i], trueLabel[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictionsArray, color = "#777777")
    plt.ylim([0,1])
    predictedLabel = np.argmax(predictionsArray)

    thisplot[predictedLabel].set_color("red")
    thisplot[trueLabel].set_color("blue")

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plotImages(i,predictions,testLabels,testImages)
plt.subplot(1,2,2)
plotValueArray(i,predictions,testLabels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plotImages(i, predictions, testLabels, testImages)
plt.subplot(1,2,2)
plotValueArray(i, predictions,  testLabels)
plt.show()

numRows = 5
numCols = 3
numImages = numRows * numCols
plt.figure(figsize = (2*2*numCols, 2*numRows))
for i in range(numImages):
    plt.subplot(numRows, 2*numCols, 2*i+1)
    plotImages(i,predictions,testLabels,testImages)
    plt.subplot(numRows,2*numCols,2*i+2)
    plotValueArray(i,predictions,testLabels)
plt.show()

img = testImages[0]
print("images shape",img.shape)

img = (np.expand_dims(img,0))
print("new images shape",img.shape)

predictionSingle = model.predict(img)
print("prediction",predictionSingle)

plotValueArray(0,predictionSingle, testLabels)
_ = plt.xticks(range(10),classNames,rotation=45)

print("prediction",np.argmax(predictionSingle[0]))
