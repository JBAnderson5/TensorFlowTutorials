#following tutorial: https://www.tensorflow.org/alpha/tutorials/sequences/word_embeddings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import io

embeddingLayers = layers.Embedding(1000, 32)

vocabSize = 10000
imdb = keras.datasets.imdb
(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words = vocabSize)

print("example review:",trainData[0])

wordIndex = imdb.get_word_index()
wordIndex = {k:(v+3) for k,v in wordIndex.items()}
wordIndex["<PAD>"] = 0
wordIndex["<START>"] = 1
wordIndex["<UNK>"] = 2
wordIndex["<UNUSED>"] = 3

reverseWordIndex = dict([(value,key) for (key,value) in wordIndex.items()])

def decodeReview(text):
    return " ".join([reverseWordIndex.get(i,"?") for i in text])

print("example review text representation>",decodeReview(trainData[0]))


maxLen = 500
trainData = keras.preprocessing.sequence.pad_sequences(trainData,value=wordIndex["<PAD>"],
    padding="post",maxlen = maxLen)
testData = keras.preprocessing.sequence.pad_sequences(testData,value=wordIndex["<PAD>"],
    padding="post",maxlen = maxLen)

print("example review padded",trainData[0])

embeddingDim = 16
model = keras.Sequential([
    layers.Embedding(vocabSize, embeddingDim, input_length = maxLen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation = "relu"),
    layers.Dense(1, activation = "sigmoid")
])
model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

history = model.fit(trainData, trainLabels, epochs = 30, batch_size = 512, validation_split = 0.2)

acc = history.history["accuracy"]
valAcc = history.history["val_accuracy"]

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (12,9))
plt.plot(epochs, acc, "bo", label = "training accuracy")
plt.plot(epochs, valAcc, "b", label = "validation accuracy")
plt.title("training and validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc = "lower right")
plt.ylim((0.5,1))
plt.show()

e = model.layers[0]
weights = e.get_weights()[0]
print("learned embeddings weight shape",weights.shape)

outV = io.open("vecs.tsv", "w", encoding = "utf-8")
outM = io.open("meta.tsv", "w", encoding = "utf-8")
for wordNum in range(vocabSize):
    word = reverseWordIndex[wordNum]
    embeddings = weights[wordNum]
    outM.write(word + "\n")
    outV.write("\t".join([str(x) for x in embeddings]) + "\n")
outV.close()
outM.close()

#can visualize the data from these two files by uploading them to: http://projector.tensorflow.org/
