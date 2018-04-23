# Work flow
We transform all image into matrices of 512x512x3. We’ve verified that all image have width of 512 pixels. For images that have height different than 512 pixels, we use Pillow library to stretch the image into 512x512. For training and validation split, we use 5% of the data for validation, and 95% for training.

### Model

We fine-tune image recognition models with pre-trained weights trained with ImageNet dataset. Our model connects the output of the original model to a dropout layer, an global average pooling layer and a Dense layer with softmax activation as output. We experience a various of image recognition models and determined that the Xception results in highest accuracy.

Loss: "categorical_crossentropy" loss for multi-class classifications.

### Training

We train using Adam optimizer with a learning of 10e-5 and save the weights of with highest accuracy on the validation set. Our model typically reach above 80% accuracy after the first batch and we stop training after the validation loss stop improving within 3 batches. As the 512x512 images takes lot of memory, we implemented the fit_generator function provided by Keras, so the training data are load on demand.


# setup

Fill `/base` dir with the data.

Run `python3 train_fasion.py` with arguments.

# arguments

```
--model_name Xception # all options: Xception, VGG16, VGG19, DenseNet121, DenseNet201, ResNet50, InceptionV3, InceptionResNetV2
--train_class_name sleeve_length_labels
--training_batch_size 64
--test_percentage 0.05
--learning_rate 0.00005
--validation_every_X_batch 5  # calculate precision, recall, f1 every X batch, set this smaller will increase speed
--memory_safe (1 or 0)
--saving_frequency 0 - 1  # set to 1 it saves every epoch, set to 0.5 it saves every 1/2 epoch
--dropout 0.2
```

