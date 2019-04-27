from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import tensorflowjs as tfjs

(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
layer1 = layers.Dense(512, activation='relu', input_shape=(28* 28,))
layer2 = layers.Dense(10, activation='softmax')
network.add(layer1)
network.add(layer2)

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

x = network.fit(train_images, train_labels, epochs=10, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)
print('test_loss', test_loss)

# network.save('my_model.h5')
tfjs.converters.save_keras_model(network, './saved_model')
