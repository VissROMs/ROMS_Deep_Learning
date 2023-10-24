# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# import torch

# Importing the Dataset
from keras.datasets import mnist
(train_images, train_labels), (test_images,test_labesl) = mnist.load_data()

# Feature scaling of train_images and test_images
train_images = train_images/255.
test_images = test_images/255.

# One Hot Encoding of train-test labels for Catecorigal cross entropy
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labesl = to_categorical(test_labesl)

# Make the images correct for conv2D
train_images = np.reshape(train_images, (60000, 28, 28, 1))
test_images = np.reshape(test_images, (10000, 28, 28, 1))

# build the model
cnn_mnist = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

cnn_mnist.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.8),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["Accuracy"]
)
history = cnn_mnist.fit(train_images, train_labels, epochs=10, batch_size=64)

D = pd.DataFrame(history.history)
[loss, accuracy] = cnn_mnist.evaluate()
