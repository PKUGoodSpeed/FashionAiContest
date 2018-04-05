import FashionTraining as FT
import FashionTransferLearning as FTT
import inception_example as INS
from keras.layers import Conv2D, Dense, Flatten, Dropout

# classes = ["collar_design_labels", "neckline_design_labels", "sleeve_length_labels", "neck_design_labels", "coat_length_labels", "lapel_design_labels", "pant_length_labels", "skirt_length_labels"]
#
# from keras.models import load_model
# fine_tuning_model = load_model('1st_version.h5')
# fine_tuning_model.layers[-1] = Dense(6, activation='softmax')
# fine_tuning_model.load_weights("weights00001820.h5")
# print(model.summary())

trainer = FTT.Trainer(train_class_name="pant_length_labels", training_batch_size=1024, learning_rate=0.005, test_percentage=0.05)
trainer.train(steps_per_epoch=64, epochs=2000)