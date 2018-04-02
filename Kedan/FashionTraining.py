import numpy as np
import csv, os, sys
from PIL import Image
import datetime

from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras import optimizers

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *

class WeightsSaver(Callback):
    def __init__(self, model, N, dir, max_copy_save=5):
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

    def __init__(self, train_class_name=None, training_batch_size=100, existing_weight=None, test_percentage=0.02, learning_rate=0.002, save_every_x_epoch=5, number_of_training_sample=sys.maxsize, fine_tuning_model=None, memory_safe=True, validate=True):

        if train_class_name == None:
            print("You must specify train_class_name")
            return

        if memory_safe == False:
            self.Y = []
            self.X = []

        self.save_every_x_epoch = save_every_x_epoch
        self.validate = validate
        self.memory_safe = memory_safe
        self.number_of_sample = number_of_training_sample
        self.train_class_name = train_class_name
        if not os.path.exists(os.path.join("models", train_class_name)):
            os.makedirs(os.path.join("models", train_class_name))

        self.training_batch_size = training_batch_size

        # We know that MNIST images are 28 pixels in each dimension.
        img_size = 512

        self.img_size_flat = img_size * img_size * 3

        self.img_shape_full = (img_size, img_size, 3)

        self.test = {}
        test_X = []
        test_Y = []
        with open('base/Annotations/label.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_class_samples = []
            for row in reader:
                if row[1] == self.train_class_name:
                    self.num_classes = len(row[2])
                    break

        # Start construction of the Keras Sequential model.
        model = Sequential()
        self.model = model

        model.add(InputLayer(input_shape=(self.img_size_flat,)))

        # The input is a flattened array with 784 elements,
        # but the convolutional layers expect images with shape (28, 28, 1)
        model.add(Reshape(self.img_shape_full))

        model.add(Reshape(self.img_shape_full))

        # First convolutional layer with ReLU-activation and max-pooling.
        model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='valid',
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
        model.add(Dense(self.num_classes, activation='softmax'))
        print(model.summary())

        self.optimizer = optimizers.Adam(lr=0.002)

        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        with open('base/Annotations/label.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_class_samples = []
            for row in reader:
                if row[1] != self.train_class_name:
                    continue
                all_class_samples.append(row)

            test_count = int(test_percentage * len(all_class_samples))
            print("Training " + train_class_name + " with: " + str(int((1 - test_percentage) * len(all_class_samples))) + ", Testing with: " + str(test_count), str(self.num_classes), "Classes")
            index = 0
            for row in all_class_samples:
                if index < len(all_class_samples) - test_count:
                    if memory_safe == False and len(self.Y) < number_of_training_sample:
                        image = Image.open("base/" + row[0])
                        img_array = np.asarray(image)
                        if img_array.shape == self.img_shape_full:
                            self.X.append(img_array.flatten())
                            self.Y.append(row[2].index("y"))
                else:
                    self.test[row[0]] = row[2].index("y")
                index += 1


            for (key, value) in self.test.items():

                image = Image.open("base/" + key)
                img_array = np.asarray(image)
                if img_array.shape != self.img_shape_full:
                    continue
                test_X.append(img_array.flatten())
                test_Y.append(value)

            self.test_X = np.array(test_X) / 255
            self.test_Y = to_categorical(test_Y, num_classes=self.num_classes)
            print(self.test_Y.shape, self.test_X.shape)

            if memory_safe == False:
                self.Y = to_categorical(self.Y, num_classes=self.num_classes)
                self.X = np.array(self.X) / 255
                print(self.X.shape, self.Y.shape)

        if existing_weight != None:
            model.load_weights(existing_weight)
        else:
            all_weights = os.listdir(os.path.join("models", self.train_class_name))
            if ".DS_Store" in all_weights:
                all_weights.remove(".DS_Store")

            if len(all_weights) > 0:
                last_copy = sorted(all_weights)[-1]
                model.load_weights(os.path.join("models", self.train_class_name, last_copy))

        if fine_tuning_model != None:
            for index in range(3):
                weights = fine_tuning_model.layers[index].get_weights()
                model.layers[index].set_weights(weights)
            print("Fine tuning model copied")

        model.save(os.path.join("models", train_class_name + "_" + "model.h5"))

    def train(self, steps_per_epoch=10, epochs=100):
        if self.validate:
            if self.memory_safe:
                self.model.fit_generator(self.generate_arrays_from(), steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=(self.test_X, self.test_Y), callbacks=[WeightsSaver(self.model, self.save_every_x_epoch, self.train_class_name)])
            else:
                self.model.fit(self.X, self.Y, batch_size=self.training_batch_size, epochs=epochs, validation_data=(self.test_X, self.test_Y), callbacks=[WeightsSaver(self.model, self.save_every_x_epoch, self.train_class_name)])
        else:
            if self.memory_safe:
                self.model.fit_generator(self.generate_arrays_from(), steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[WeightsSaver(self.model, self.save_every_x_epoch, self.train_class_name)])
            else:
                self.model.fit(self.X, self.Y, batch_size=self.training_batch_size, epochs=epochs, callbacks=[WeightsSaver(self.model, self.save_every_x_epoch, self.train_class_name)])

    def generate_arrays_from(self):
        Y = []
        X = []
        while 1:
            with open('base/Annotations/label.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                index = 1
                for row in reader:
                    if row[1] != self.train_class_name:
                        continue
                    if row[0] in self.test:
                        continue
                    image = Image.open("base/" + row[0])
                    img_array = np.asarray(image)
                    if img_array.shape != self.img_shape_full:
                        continue
                    X.append(img_array.flatten())
                    Y.append(row[2].index("y"))
                    if index % self.training_batch_size == 0:
                        x, y = np.array(X) / 255, to_categorical(Y, num_classes=self.num_classes)
                        yield (x, y)
                        Y = []
                        X = []
                    if index > self.number_of_sample:
                        break
                    index += 1




