#### Modules
1. numpy 1.11.3
2. pandas 0.18.1
3. skimage 0.12.3

#### Usage:

The Image preprocessing logics are in `process.py`, in the class `ImagePrec`:
  
  1. First load all images into memory, and all of them are resized into squres (e.g. 256 x 256). For rectangle images are padded into squares using `_pad_square(img)` before resized.
  2. Training and validation sets are generated with `getbatch(self, idx, reflect=False, random_crop=0, crop_resize=False)` by passing in a list of indices. The augmentations include left-right reflection and random crops.
  3. When the training and validating finished, it loads the testing data and generate a testing set to make predictions.

##### A brief illustration is shown in `test_img.ipynb`
