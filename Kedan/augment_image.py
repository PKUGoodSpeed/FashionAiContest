import csv
from PIL import Image
import numpy as np

X = []
index = 0
with open('base/Annotations/label.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    all_class_samples = []
    for row in reader:
        image = Image.open("base256/" + row[0].replace("/", "-"))

        for w in range(3):
            img2 = image.crop((w*16, 0, 224 + w*16, 256))
            img2 = img2.resize((224, 224), Image.ANTIALIAS)
            img2.save("base224augmented/" + row[0].replace("/", "-")[:-4] + "~" + str(w)  + ".jpg", "JPEG", quality=224, optimize=True, progressive=True)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img2.save("base224augmented/" + row[0].replace("/", "-")[:-4] + "~" + str(w) + "~flip" + ".jpg", "JPEG", quality=224, optimize=True, progressive=True)

        index += 1
        if index % 500 == 0:
            print(index)