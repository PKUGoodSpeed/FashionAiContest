from stage1 import DFA_full_stage1
import tensorflow as tf
import csv
from PIL import Image
import numpy as np
import os
name = []
X = []
index = 0

for img in os.listdir("base224augmented/"):
    if img[-3:] == "jpg":
        image = Image.open("base224augmented/" + img)
        name.append("FashionEmbeddings-augmented/" + img)
        img_array = np.asarray(image).astype("float32")
        X.append(img_array)
        index += 1
        if index % 200 == 0:
            print(index)
            X = np.array(X)
            tf.reset_default_graph()
            net = DFA_full_stage1({'data': X})

            sesh = tf.Session()
            # Load the data
            net.load('stage1.npy', sesh, ignore_missing=True)
            # Forward pass
            output = sesh.run(net.get_output())
            rest = list(output)
            for i in range(len(rest)):
                np.save(name[i][:-3] + "npy", rest[i])
            X = []
            name =[]
