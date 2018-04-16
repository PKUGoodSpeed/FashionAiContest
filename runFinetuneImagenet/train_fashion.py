import XceptionTraining as XT

import argparse, sys

parser = argparse.ArgumentParser()

parser.add_argument('--train_class_name', default="sleeve_length_labels")
parser.add_argument('--training_batch_size', default=64)
parser.add_argument('--learning_rate', default=0.00005)
parser.add_argument('--test_percentage', default=0.05)

args = parser.parse_args()

print(args)
trainer = XT.Trainer(train_class_name=args.train_class_name, training_batch_size=int(args.training_batch_size), learning_rate=float(args.learning_rate), test_percentage=float(args.test_percentage))
trainer.train(steps_per_epoch=64, epochs=5000)