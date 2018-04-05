from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input


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

cnn_model = InceptionResNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
inputs = Input((224, 224, 3))

x = inputs
x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)

model = Model(inputs, x)

for img in os.listdir("base224/"):
    if img[-3:] == "jpg":
        image = Image.open("base224/" + img)
        img_array = np.asarray(image).astype("float32")
        print(img_array.shape)
        result = model.predict(np.array(img_array))
        print(result.shape)





