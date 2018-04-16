import XceptionTraining as XT

trainer = XT.Trainer(train_class_name="sleeve_length_labels", training_batch_size=32, learning_rate=0.00005, test_percentage=0.05, save_every_x_epoch=10)
trainer.train(steps_per_epoch=64, epochs=5000)