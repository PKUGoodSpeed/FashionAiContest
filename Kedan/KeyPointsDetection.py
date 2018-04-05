import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv, os, sys
import math
from PIL import Image
import datetime
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from tensorflow.python.keras import optimizers
from keras import backend as K
from keras import metrics

class WeightsSaver(Callback):
    def __init__(self, model, N, dir, max_copy_save=10):
        self.model = model
        self.N = N
        self.batch = 0
        self.max_copy_save = max_copy_save
        self.dir = dir

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = os.path.join("models", self.dir, "{date:%Y-%m-%d %H:%M:%S}-".format( date=datetime.datetime.now()) + 'weights%08d.h5' % self.batch)
            self.model.save_weights(name)
            all_weights = os.listdir(os.path.join("models", self.dir))
            if ".DS_Store" in all_weights:
                all_weights.remove(".DS_Store")

            if len(all_weights) > self.max_copy_save:
                first_copy = sorted(all_weights)[0]
                os.remove(os.path.join("models", self.dir, first_copy))
        self.batch += 1

class Trainer:

    def __init__(self, training_batch_size=100, existing_weight=None, test_percentage=0.02, learning_rate=0.002, save_every_x_epoch=5, number_of_training_sample=sys.maxsize, fine_tuning_model=None, memory_safe=True, validate=True):

        if memory_safe == False:
            self.Y = []
            self.X = []

        self.save_every_x_epoch = save_every_x_epoch
        self.validate = validate
        self.memory_safe = memory_safe
        self.number_of_sample = number_of_training_sample
        self.training_batch_size = training_batch_size

        # We know that MNIST images are 28 pixels in each dimension.
        self.img_size = 160

        self.img_size_flat = self.img_size * self.img_size * 3

        self.img_shape_full = (self.img_size, self.img_size, 3)

        self.test = {}
        test_X = []
        test_Y = []
        with open('train_labels.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_class_samples = []
            for row in reader:
                all_class_samples.append(row)

            print(len(all_class_samples))
            test_count = int(test_percentage * len(all_class_samples))
            print("Training with: " + str(int((1 - test_percentage) * len(all_class_samples))) + ", Testing with: " + str(test_count))
            index = 0
            for row in all_class_samples:

                y = []
                for index in range(1, 7):
                    if index < 3:
                        y.append(float(row[index]))
                    else:
                        y.append(float(row[index]) / 160)
                print(y)

                if index > test_count:
                    if memory_safe == False and len(self.Y) < number_of_training_sample:
                        image = Image.open(row[0])
                        img_array = np.asarray(image)
                        if img_array.shape != self.img_shape_full:
                            continue
                        self.X.append(img_array.flatten())
                        self.Y.append(y)
                        continue
                    else:
                        break

                self.test[row[0]] = y
                index += 1

            for (key, value) in self.test.items():

                image = Image.open(key)
                img_array = np.asarray(image)
                if img_array.shape != self.img_shape_full:
                    continue
                test_X.append(img_array.flatten())
                test_Y.append(value)

            self.test_X = np.array(test_X) / 255
            self.test_Y = np.array(test_Y)

            if memory_safe == False:
                self.Y = np.array(self.Y)
                self.X = np.array(self.X) / 255
                print(self.X.shape, self.Y.shape)

        # Start construction of the Keras Sequential model.
        model = Sequential()
        self.model = model

        model.add(InputLayer(input_shape=(self.img_size_flat,)))

        # The input is a flattened array with 784 elements,
        # but the convolutional layers expect images with shape (28, 28, 1)
        model.add(Reshape(self.img_shape_full))

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

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(6))

        def normalize_loss(yTrue, yPred):
            total_loss = 0
            total_loss += K.sum(K.square(yTrue[0:2] - yPred[0:2]))
            # if yTrue[0] == 1:
            #     total_loss += K.sum(K.square(yTrue[3:5] - yPred[3:5]))
            if yTrue[0] == 1:
                total_loss += K.sum(K.square(yTrue[2:4] - yPred[2:4]))
            else:
                total_loss += K.sum(K.square(yTrue[0] - yPred[0]))

            if yTrue[1] == 1:
                total_loss += K.sum(K.square(yTrue[4:6] - yPred[4:6]))
            else:
                total_loss += K.sum(K.square(yTrue[1] - yPred[1]))

            return K.sqrt(total_loss)

        self.optimizer = optimizers.Adadelta(lr=learning_rate, clipnorm=2.)

        model.compile(optimizer=self.optimizer, loss=normalize_loss, metrics=[metrics.cosine,  metrics.mse])
        if existing_weight != None:
            model.load_weights(existing_weight)
        else:
            all_weights = os.listdir(os.path.join("models", "keypoint_model"))
            if ".DS_Store" in all_weights:
                all_weights.remove(".DS_Store")

            if len(all_weights) > 0:
                last_copy = sorted(all_weights)[-1]
                model.load_weights(os.path.join("models", "keypoint_model", last_copy))

        if fine_tuning_model != None:
            for index in range(3):
                weights = fine_tuning_model.layers[index].get_weights()
                model.layers[index].set_weights(weights)
            print("Fine tuning model copied")

        model.save(os.path.join("models", "keypoint_model.h5"))

    def train(self, epochs=100):
        if self.validate:
            self.model.fit(self.X, self.Y, batch_size=self.training_batch_size, epochs=epochs, validation_data=(self.test_X, self.test_Y), callbacks=[WeightsSaver(self.model, self.save_every_x_epoch, "keypoint_model")])
        else:
            self.model.fit(self.X, self.Y, batch_size=self.training_batch_size, epochs=epochs, callbacks=[WeightsSaver(self.model, self.save_every_x_epoch, "keypoint_model")])

    def classify(self, file):
        image = Image.open(file)
        img_array = np.asarray(image)
        if img_array.shape != self.img_shape_full:
            return
        X = np.array([img_array.flatten()]) / 255
        result = self.model.predict(X)
        return result


