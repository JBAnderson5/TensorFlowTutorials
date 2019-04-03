#following tutorial: https://www.tensorflow.org/alpha/tutorials/keras/save_and_restore_models

import tensorflow as tf
from tensorflow import keras
import os

(trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.mnist.load_data()
numDataPoints = 10000
trainLabels = trainLabels[:numDataPoints]
testLabels = testLabels[:numDataPoints]
trainImages = trainImages[:numDataPoints].reshape(-1,28*28) / 255.0
testImages = testImages[:numDataPoints].reshape(-1,28*28) / 255.0

def createModel():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation = "relu", input_shape = (784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation = "softmax")
    ])
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    return model

model = createModel()
model.summary()

checkpointPath = "training_1/cp.ckpt"
checkpointDir = os.path.dirname(checkpointPath)

#just saves weights
cpCallback = tf.keras.callbacks.ModelCheckpoint(checkpointPath,save_weights_only = True, verbose = 1)

model = createModel()
model.fit(trainImages, trainLabels, epochs = 10, validation_data = (testImages,testLabels),callbacks = [cpCallback])

model = createModel()

loss, acc = model.evaluate(testImages, testLabels)
print("untrained model, accuracy {:5.2f}%".format(100*acc))

model.load_weights(checkpointPath)
loss, acc = model.evaluate(testImages, testLabels)
print("restored model, accuracy {:5.2f}%".format(100*acc))

checkpointPath = "training_2/cp-{epoch:04d}.ckpt"
checkpointDir = os.path.dirname(checkpointPath)

cpCallback = tf.keras.callbacks.ModelCheckpoint(checkpointPath, verbose = 1, save_weights_only = True, period = 5)
model = createModel()
model.save_weights(checkpointPath.format(epoch=0))
model.fit(trainImages, trainLabels, epochs = 50, callbacks = [cpCallback],
    validation_data = (testImages,testLabels), verbose = 0)

latest = tf.train.latest_checkpoint(checkpointDir)
print(latest)

model = createModel()
model.load_weights(latest)
loss, acc = model.evaluate(testImages, testLabels)
print("restored model, accuracy {:5.2f}%".format(100*acc))

#just saves weights of model. need to load the weights into a model with the same architecture as original
model.save_weights("./checkpoints/my_checkpoint")
model = createModel()
model.load_weights("./checkpoints/my_checkpoint")
loss,acc = model.evaluate(testImages, testLabels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#saves entire model as an HDF5 file
model = createModel()
model.fit(trainImages, trainLabels, epochs = 5)
model.save("my_model.h5")

newModel = keras.models.load_model("my_model.h5")
newModel.summary()

loss, acc = newModel.evaluate(testImages,testLabels)
print("restored model, accuracy: {:5.2f}%".format(100*acc))


#saves entire model as a saved_model. note this is experimental and might change
model = createModel()
model.fit(trainImages,trainLabels, epochs = 5)

import time
savedModelPath = "./savedModels/{}".format(int(time.time()))
tf.keras.experimental.export_saved_model(model, savedModelPath)
print(savedModelPath)

newModel = tf.keras.experimental.load_from_saved_model(savedModelPath)
newModel.summary()

print(model.predict(testImages).shape)

newModel.compile(optimizer = model.optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
loss, acc = newModel.evaluate(testImages,testLabels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
