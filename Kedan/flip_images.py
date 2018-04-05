import csv
from PIL import Image
import numpy as np
import os

X = []
index = 0
for img in os.listdir("base224/"):
    if img[-3:] == "jpg":
        image = Image.open("base224/" + img)
        img2 = image.transpose(Image.FLIP_LEFT_RIGHT)
        img2.save("base224flip/" + img, "JPEG", quality=224, optimize=True, progressive=True)

    index += 1
    if index % 500 == 0:
        print(index)