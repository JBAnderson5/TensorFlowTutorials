#following tutorial: https://www.tensorflow.org/alpha/tutorials/images/intro_to_cnns

from tensorflow.keras import datasets, layers, models

(trainImages, trainLabels), (testImages, testLabels) = datasets.mnist.load_data()
trainImages = trainImages.reshape((60000,28,28,1))
testImages = testImages.reshape((10000,28,28,1))

#normalizes values to be between 0 and 1
trainImages, testImages = trainImages/255.0, testImages/255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = "relu", input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10, activation = "softmax"))

model.summary()

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(trainImages, trainLabels, epochs = 5)

testLoss, testAcc = model.evaluate(testImages,testLabels)

print("test accuracy:",testAcc)
