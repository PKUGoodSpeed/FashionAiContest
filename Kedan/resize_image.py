import csv
from PIL import Image
import numpy as np

X = []
index = 0
with open('base/Annotations/label.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    all_class_samples = []
    for row in reader:
        image = Image.open("base/" + row[0])
        image = image.resize((256, 256), Image.ANTIALIAS)
        image.save("base256/" + row[0].replace("/", "-"), "JPEG", quality=256, optimize=True, progressive=True)
        index += 1
        if index % 500 == 0:
            print(index)