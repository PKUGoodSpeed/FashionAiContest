import KeyPointsDetection as KPD
import os
from PIL import Image, ImageDraw

prediction_thresh = 0.5
image_size = 160
r = 2

trainer = KPD.Trainer(training_batch_size=128, learning_rate=0.002, number_of_training_sample=256, fine_tuning_model=None, memory_safe=False, validate=False, test_percentage=0)
trainer.train(epochs=200)

for file in os.listdir("test"):
    if file != ".DS_Store":
        result = trainer.classify("test/"+file)[0]

        image = Image.open("test/"+file)
        draw = ImageDraw.Draw(image)

        if result[0] > prediction_thresh:
            draw.ellipse((image_size * result[2] - r, image_size * result[3] - r, image_size * result[2] + r, image_size * result[3] + r), fill=(255, 0, 0, 0))

        if result[1] > prediction_thresh:
            draw.ellipse((image_size * result[4] - r, image_size * result[5] - r, image_size * result[4] + r,
                          image_size * result[5] + r), fill=(0, 255, 0, 0))

        # if result[2] > prediction_thresh:
        #     draw.ellipse((image_size * result[7] - r, image_size * result[8] - r, image_size * result[7] + r,
        #                   image_size * result[8] + r), fill=(0, 0, 255, 0))

        image.save("test_result/"+file)