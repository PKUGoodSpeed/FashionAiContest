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
```

