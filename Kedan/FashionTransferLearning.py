import numpy as np
import csv, os, sys
from PIL import Image
import datetime

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import Adam, SGD, Adadelta
from keras.regularizers import l2
import keras

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

    def __init__(self, train_class_name=None, training_batch_size=100, load_weight=True, existing_weight=None, test_percentage=0.02, learning_rate=0.002, save_every_x_epoch=5, number_of_training_sample=sys.maxsize):

        if train_class_name == None:
            print("You must specify train_class_name")
            return

        self.Y = []
        self.X = []
        self.save_every_x_epoch = save_every_x_epoch
        self.number_of_sample = number_of_training_sample
        self.train_class_name = train_class_name
        if not os.path.exists(os.path.join("models", train_class_name)):
            os.makedirs(os.path.join("models", train_class_name))

        self.training_batch_size = training_batch_size

        with open('base/Annotations/label.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_class_samples = []
            for row in reader:
                if row[1] == self.train_class_name:
                    self.num_classes = len(row[2])
                    break

        model = Sequential()
        model.add(InputLayer(input_shape=(7, 7, 512,)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu', kernel_initializer='random_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.))
        model.add(Dense(self.num_classes, activation='softmax'))
        print(model.summary())
        self.optimizer = Adadelta(lr=learning_rate)
        self.model = model
        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        with open('base/Annotations/label.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_class_samples = []
            for row in reader:
                if row[1] != self.train_class_name:
                    continue
                all_class_samples.append(row)

            for row in all_class_samples:
                np_file = "FashionEmbeddings/" + row[0].replace("/", "-")[:-3] + "npy"
                if os.path.exists(np_file):
                    np_row = np.load(np_file)
                    self.X.append(np_row)
                    self.Y.append(row[2].index("y"))

                np_file = "FashionEmbeddings-flip/" + row[0].replace("/", "-")[:-3] + "npy"
                if os.path.exists(np_file):
                    np_row = np.load(np_file)
                    self.X.append(np_row)
                    self.Y.append(row[2].index("y"))

            self.Y = to_categorical(self.Y, num_classes=self.num_classes)
            self.X = np.array(self.X)
            print(self.X.shape, self.Y.shape)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = test_percentage, random_state=42)

        if load_weight:
            if existing_weight != None:
                model.load_weights(existing_weight)
            else:
                all_weights = os.listdir(os.path.join("models", self.train_class_name))
                if ".DS_Store" in all_weights:
                    all_weights.remove(".DS_Store")

                if len(all_weights) > 0:
                    last_copy = sorted(all_weights)[-1]
                    model.load_weights(os.path.join("models", self.train_class_name, last_copy))
                    y_prob = model.predict(self.X_test)
                    y_classes = y_prob.argmax(axis=-1)
                    print(self.y_test.shape, y_classes.shape)
                    print(classification_report(self.y_test, to_categorical(y_classes, num_classes=self.num_classes)))
        model.save(os.path.join("models", train_class_name + "_" + "model.h5"))

    def train(self, steps_per_epoch=10, epochs=100):
        self.model.fit(self.X_train, self.y_train, batch_size=self.training_batch_size, epochs=epochs,
                       validation_data=(self.X_test, self.y_test),
                       callbacks=[WeightsSaver(self.model, self.save_every_x_epoch, self.train_class_name)])

