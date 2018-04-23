# benchmarks
Here is the current training results so far:


| LABELS                                     | resnet50    | inception   | xception    | stacking               |
| ------------------------------------------ | ----------- | ----------- | ----------- | ---------------------- |
| neck_design_label (5696 images)            | ~78% (256)  | ~80.4 (256) | ~81% (256)  | 83.2%                  |
| collar_design_labels (8393 images)         | ~81% (256)  | ~82% (256)  | ~84% (256)  | 86.7%                  |
| lapel_design_labels (7034 images)          | ~84% (256)  | ~84% (256)  | ~86% (256)  | 88.92%                 |
| neckline_design_labels (17148 images)      | ~80% (256)  | ~81% (256)  | ~79% (256)  | 83.68%                 |
| sleeve_length_labels (13299 images)        | ~84% (256)  | ~87% (256)  | ~88% (256)  | 88.3%                  |
| coat_length_labels (11320 images)          | 80.2% (256) | 84.7% (256) | 85.6% (256) | 84.6% (using 128 imgs) |
| pant_length_labels (7460 images) @Kedan L  |             |             | 0.87 (299)  |                        |
| skirt_length_labels (9223 images) @Kedan L |             |             |             |                        |


coat_length_labels (11320 images):
resnet50: 80% (128)
xception: 81% (128)
inception: 76%(128)
vgg16: 79%(128)
vgg19:76.6% (128)
customized resnet: 67% (128) (Too much dropout)

512 x 512

| LABELS                                | Xception                        | DenseNet121     | DenseNet201       | ResNet50 | VGG16 | VGG19 | InceptionResNetV2 | InceptionV3  |
| ------------------------------------- | ------------------------------- | --------------- | ----------------- | -------- | ----- | ----- | ----------------- | ------------ |
| neck_design_label (5696 images)       | 0.912 (9/11 epoch)
0.89(epoch7) |                 |                   |          |       |       |                   |              |
| collar_design_labels (8393 images)    | 87 (5 epoch)                    |                 |                   |          |       |       |                   |              |
| lapel_design_labels (7034 images)     | 90% (15 epoch)                  |                 |                   |          |       |       |                   |              |
| neckline_design_labels (17148 images) |                                 |                 |                   |          |       |       |                   | 85 (3 epoch) |
| sleeve_length_labels (13299 images)   |                                 |                 | 86.5% (1.5 epoch) |          |       |       |                   |              |
| coat_length_labels (11320 images)     | 87.5% (9 epoch)                 |                 |                   |          |       |       |                   |              |
| pant_length_labels (7460 images)      | 88% (5 epoch)                   | 86.5% (2 epoch) |                   |          |       |       |                   |              |
| skirt_length_labels (9223 images)     | 89% (9 epoch)                   |                 |                   |          |       |       |                   |              |


@Kedan L I will not run neckline_design_labels again, since the data amount is too large.

- New test data: https://drive.google.com/file/d/19aIvVsAeCZAso5ljEoUl-aGeEYCouHin/view?usp=sharing @doc (fashionAi_attributes_test_b_20180418.tar)
- Prediction data (or together with validation set prediction) uploaded here: https://drive.googl e.com/drive/folders/1NO6vHluF4ByD0wrW9mCPKm6xseuUXtjN?usp=sharing @doc 
  - Filenaming: MODEL_LABEL_SIZE_TYPE.csv or .pik @doc 
    - MODEL: resnet/ xception/ inception/ vgg …
    - LABEL: neck_design_labels/ pant_length_labels ..
    - SIZE: 256/ 512 …
    - TYPE: valid/ pred
    - format: pickle or csv
    - dataframe columns: fname, label, prediction (the same as submission)


**Stack status:**

| LABELS                                | MERGE STATUS                              |
| ------------------------------------- | ----------------------------------------- |
| neck_design_label (5696 images)       | full                                      |
| collar_design_labels (8393 images)    | full                                      |
| lapel_design_labels (7034 images)     | full                                      |
| neckline_design_labels (17148 images) | part (not gonna merge this one)           |
| sleeve_length_labels (13299 images)   | full                                      |
| coat_length_labels (11320 images)     | part (cannot be used for stack right now) |
| pant_length_labels (7460 images)      | part (cannot merge so far, running)       |
| skirt_length_labels (9223 images)     | ready                                     |



