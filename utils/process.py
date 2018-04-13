'''
Preprocessing code for FashionAiContest
'''
# Basic
import os
import time
import numpy as np
import pandas as pd

# Image processing
from skimage.io import imread, imshow, concatenate_images, imsave
from skimage.morphology import label
from skimage.transform import resize
from skimage.util import pad

# Using multiprocessing
from multiprocessing import Pool

# Visualization for checking results
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def _str2cls(s):
    ''' Converting the labeling string into one hot classes '''
    cls = np.array([c for c in s])
    prob = 1.*(cls == 'y').astype(np.float32) + 0.5*(cls == 'm').astype(np.float32)
    assert prob.sum() >= 1., "The {STRING} is not a valid label.".format(STRING=s)
    return prob / prob.sum()

def _pad_square(img):
    ''' Padding rectangle images into squares '''
    h, w = img.shape[: 2]
    L = max(h, w)
    left = int((L-w)/2)
    right = L-w-left
    top = int((L-h)/2)
    bottom = L-h-top
    return pad(img, ((top, bottom), (left, right), (0, 0)), 'linear_ramp')

def _crop_center(img, margin=0.1):
    ''' Randomly do a crop on the center part of the image '''
    h, w = img.shape[: 2]
    h_, w_ = int(h*(1-margin)), int(w*(1-margin))
    i = np.random.randint(h-h_+1)
    j = np.random.randint(w-w_+1)
    return img[i: i+h_, j: j+w_]

def _crop_center_resize(img, shape, margin=0.1):
    ''' Resize after crop'''
    return resize(_crop_center(img, margin), shape, mode='edge', preserve_range=True)

class ImagePrec:
    _labels = None
    _imgs = None
    _df = None
    _input_shape = None
    _output_dim = None
    
    def __init__(self, category, label_file, img_path, pad_square=False, size=None):
        start_time = time.time()
        df = pd.read_csv(label_file, names=['fname', 'category', 'class'])
        assert category in set(df['category'].values), "Wrong category name."
        df = df[df['category'] == category]
        self._df = df
        self._imgs = []
        self._labels = []
        for fname, label in zip(df['fname'].tolist(), df['class'].tolist()):
            img = imread(img_path+'/'+fname)/255.
            assert img.shape[2] == 3, "There are images having other than 3 channels."
            if pad_square:
                img = _pad_square(img)
            if size is not None:
                assert type(size) == int, "The size of images should be integer."
                img = resize(img, (size, size), mode='edge', preserve_range=True)
            self._imgs.append(img)
            self._labels.append(label)
        self._input_shape = (size, size, 3)
        self._output_dim = len(df['class'].tolist()[0])
        print("Time usage for loading the images is {TIME} sec.".format(TIME=str(time.time() - start_time)))
    
    def getbatch(self, idx, reflect=False, random_crop=0, crop_resize=False):
        ''' 
        Getting batches for training or validation
        reflect: whether add left-right reflection
        random_crop: number of random_crop on that images that will be added into the data set
        '''
        start_time = time.time()
        x = [self._imgs[i] for i in idx]
        y = [_str2cls(self._labels[i]) for i in idx]
        n_base = len(idx)
        if reflect:
            x += [np.fliplr(img) for img in x]
            y += y
        for i in range(n_base):
            img = x[i]
            label = y[i]
            for _ in range(random_crop):
                y.append(label)
                if crop_resize:
                    shape = img.shape[:2]
                    x.append(_crop_center_resize(img, shape, margin=0.1))
                else:
                    x.append(_crop_center(img, margin=0.1))
        print("Time usage for generating batch is {TIME} sec.".format(TIME=str(time.time() - start_time)))
        return np.array(x), np.array(y)

    def getDataFrame(self):
        """ Getting DataFrame for particular labels """
        return self._df

    def getInputShape(self):
        return self._input_shape

    def getOutputDim(self):
        return self._output_dim

    def getTests(self, category, label_file, img_path, pad_square=False, size=None):
        """ Used when making predictions for the test case """
        start_time = time.time()
        df = pd.read_csv(label_file, names=['fname', 'category', 'class'])
        assert category in set(df['category'].values), "Wrong category name."
        df = df[df['category'] == category]
        self._df = df
        self._imgs = []
        self._labels = []
        for fname, label in zip(df['fname'].tolist(), df['class'].tolist()):
            img = imread(img_path+'/'+fname)/255.
            assert img.shape[2] == 3, "There are images having other than 3 channels."
            if pad_square:
                img = _pad_square(img)
            if size is not None:
                assert type(size) == int, "The size of images should be integer."
                img = resize(img, (size, size), mode='edge', preserve_range=True)
            self._imgs.append(img)
            self._labels.append(label)
        self._input_shape = (size, size, 3)
        print("Time usage for loading the Test images is {TIME} sec.".format(TIME=str(time.time() - start_time)))

    def getTestBatch(self):
        ''' 
        Getting test batches for testing purpose
        '''
        return np.array(self._imgs)


if __name__ == '__main__':
    fname = "../data/base/Annotations/label.csv"
    path = "../data/web"
    category = "collar_design_labels"
    ip = ImagePrec(category=category, label_file=fname, img_path=path,
    pad_square=False, size=None)
    ip.pad()