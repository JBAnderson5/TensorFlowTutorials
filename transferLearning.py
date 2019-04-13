#following tutorial: https://www.tensorflow.org/alpha/tutorials/images/transfer_learning
"""
the general machine learning workflow.

1. Examine and understand the data
2. Build an input pipeline, in this case using Keras ImageDataGenerator
3. Compose our model
    * Load in our pretrained base model (and pretrained weights)
    * Stack our classification layers on top
4. Train our model
5. Evaluate model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds

SPLIT_WEIGHTS = (8,1,1)
splits = tfds.Split.TRAIN.subsplit(weighted = SPLIT_WEIGHTS)

(rawTrain,rawValidation,rawTest), metadata = tfds.load("cats_vs_dogs",split=list(splits),with_info=True,as_supervised=True)

print(metadata)
print(rawTrain)
print(rawValidation)
print(rawTest)

getLabelName = metadata.features["label"].int2str

for image,label in rawTrain.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(getLabelName(label))
plt.show()

IMG_SIZE = 160

def formatExample(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE,IMG_SIZE))
    return image, label

train = rawTrain.map(formatExample)
validation = rawValidation.map(formatExample)
test = rawTest.map(formatExample)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

trainBatches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validationBatches = validation.batch(BATCH_SIZE)
testBatches = test.batch(BATCH_SIZE)

for imageBatch, labelBatch in trainBatches.take(1):
    pass
print("image batch shape:", imageBatch.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

baseModel = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = "imagenet")
featureBatch = baseModel(imageBatch)
print("feature batch shape:", featureBatch.shape)

baseModel.trainable = False
baseModel.summary()

globalAverageLayer = tf.keras.layers.GlobalAveragePooling2D()
featureBatchAverage = globalAverageLayer(featureBatch)
print("feature batch average shape:",featureBatchAverage.shape)

predictionLayer = keras.layers.Dense(1)
predictionBatch = predictionLayer(featureBatchAverage)
print("prediction batch shape:",predictionBatch.shape)

model = tf.keras.Sequential([
    baseModel,
    globalAverageLayer,
    predictionLayer
])

baseLearningRate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = baseLearningRate),
    loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()

print("number of layers with trainiable variables:", len(model.trainable_variables))

numTrain, numVal, numTest = (
    metadata.splits["train"].num_examples*weight/10
    for weight in SPLIT_WEIGHTS)

initialEpochs = 10
stepsPerEpoch = round(numTrain)//BATCH_SIZE
validationSteps = 20

loss0,accuracy0 = model.evaluate(validationBatches, steps = validationSteps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(trainBatches,epochs = initialEpochs, validation_data = validationBatches)

acc = history.history["accuracy"]
valAcc = history.history["val_accuracy"]

loss = history.history["loss"]
valLoss = history.history["val_loss"]

plt.figure(figsize = (8,8))
plt.subplot(2, 1, 1)
plt.plot(acc, label = "training Accuracy")
plt.plot(valAcc, label = "validation Accuracy")
plt.legend(loc = "lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()),1])
plt.title("training and validation accuracy")

plt.subplot(2,1,2)
plt.plot(loss, label = "training loss")
plt.plot(valLoss, label = "validation loss")
plt.legend(loc = "upper right")
plt.ylabel("cross entropy")
plt.ylim([0,1.0])
plt.title("training and validation loss")
plt.xlabel("epoch")
plt.show()

baseModel.trainable = True
print("number of layers in base model:",len(baseModel.layers))
fineTuneAt = 100
for layer in baseModel.layers[:fineTuneAt]:
    layer.trainable = False

model.compile(loss = "binary_crossentropy", optimizer = tf.keras.optimizers.RMSprop(lr=baseLearningRate/10),
    metrics = ["accuracy"])

model.summary()

print("number of layers with trainable variables", len(model.trainable_variables))

fineTuneEpochs = 10
totalEpochs = initialEpochs + fineTuneEpochs
historyFine = model.fit(trainBatches, epochs = totalEpochs,
    initial_epoch = initialEpochs, validation_data = validationBatches)

acc += historyFine.history["accuracy"]
valAcc += historyFine.history["val_accuracy"]
loss += historyFine.history["loss"]
valLoss += historyFine.history["val_loss"]

plt.figure(figsize = (8,8))
plt.subplot(2, 1, 1)
plt.plot(acc, label = "training Accuracy")
plt.plot(valAcc, label = "validation Accuracy")
plt.ylabel("Accuracy")
plt.ylim([0.8,1])
plt.plot([initialEpochs - 1,initialEpochs - 1], plt.ylim(), label = "start fine tuning")
plt.legend(loc = "lower right")
plt.title("training and validation accuracy")

plt.subplot(2,1,2)
plt.plot(loss, label = "training loss")
plt.plot(valLoss, label = "validation loss")

plt.ylabel("cross entropy")
plt.ylim([0,1.0])
plt.plot([initialEpochs - 1,initialEpochs - 1], plt.ylim(), label = "start fine tuning")
plt.legend(loc = "upper right")
plt.title("training and validation loss")
plt.xlabel("epoch")
plt.show()
