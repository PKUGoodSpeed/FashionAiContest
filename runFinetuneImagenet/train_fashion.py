import XceptionTraining as XT
import XceptionTrainingMemorySafe as XTMS

import argparse, sys

parser = argparse.ArgumentParser()

parser.add_argument('--train_class_name', default="skirt_length_labels")
parser.add_argument('--training_batch_size', default=64)
parser.add_argument('--learning_rate', default=0.00005)
parser.add_argument('--test_percentage', default=0.05)
parser.add_argument('--memory_safe', default=0)
parser.add_argument('--validation_every_X_batch', default=5)

args = parser.parse_args()

print(args)

if int(args.memory_safe) == 0:
	trainer = XT.Trainer(train_class_name=args.train_class_name, training_batch_size=int(args.training_batch_size), learning_rate=float(args.learning_rate), test_percentage=float(args.test_percentage), validation_every_X_batch=int(args.validation_every_X_batch))
	trainer.train(steps_per_epoch=64, epochs=5000)
else:
	trainer = XTMS.Trainer(train_class_name=args.train_class_name, training_batch_size=int(args.training_batch_size), learning_rate=float(args.learning_rate), test_percentage=float(args.test_percentage))
	trainer.train(epochs=5000)
