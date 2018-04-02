import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv, os
import math
from PIL import Image

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from tensorflow.python.keras.optimizers import Adam

img_size = 512

img_size_flat = img_size * img_size * 3

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 3)

# Number of classes, one class for each of 10 digits.
num_classes = 6


model = Sequential()

model.add(InputLayer(input_shape=(img_size_flat,)))

# The input is a flattened array with 784 elements,
# but the convolutional layers expect images with shape (28, 28, 1)
model.add(Reshape(img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=2, filters=64, padding='same',
                 activation='relu', name='layer_conv3'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# First fully-connected / dense layer with ReLU-activation.
model.add(Dense(128, activation='relu'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=0.002)


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1

print("Compile")
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights("weights00001768.h5")
print(model.metrics_names)

print("model loaded")

Y = []
X = []

with open('base/Annotations/label.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    index = 1
    for row in reader:
        if row[1] != "skirt_length_labels":
            continue
        image = Image.open("base/" + row[0])
        img_array = np.asarray(image)
        if img_array.shape != (img_size, img_size, 3):
            continue
        X.append(img_array.flatten())
        Y.append(row[2].index("y"))
        if index % 1000 == 0:
            x, y = np.array(X) / 255, to_categorical(Y, num_classes=num_classes)
            score = model.evaluate(x, y)
            print(score)

            Y = []
            X = []
            print(index)
        index += 1