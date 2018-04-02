from cascade import DFA_full_cascade
from stage1 import DFA_full_stage1
import tensorflow as tf
import csv
from PIL import Image
import numpy as np
import os

name = []
X = []
index = 0
with open('base/Annotations/label.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    all_class_samples = []
    for row in reader:

        if row[1] == "coat_length_labels":
            image = Image.open("base224/" + row[0].replace("/", "-"))
            if os.path.exists("FashionEmbeddings3") == False:
                os.makedirs("FashionEmbeddings3")
            name.append("FashionEmbeddings3/" + row[0].replace("/", "-"))
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
                print(output.shape)
                rest = list(output)
                for i in range(len(rest)):
                    np.save(name[i][:-3] + "npy", rest[i])
                X = []
                name =[]
