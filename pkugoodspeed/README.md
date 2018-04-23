## Environment
- GPU: Tesla K80
- python2.7 module:
  - numpy 1.11.3
  - pandas 0.18.1
  - tensorflow 1.3.0
  - keras: 2.0.2
  - pickle: ['1.0', '1.1', '1.2', '1.3', '2.0']
- Use "../utils/opts_parser.py" to load config files.
- Training and Testing data should be put in the folder "../data"


## Models:
Model implementations are in `models`.
We tried: `naive CNNs`, `vgg16`, `vgg19`, `Resnet50`, `inception`, and `xception`, among which the `xception` net perform best for most of the labels.

Loss: "categorical_crossentropy" loss for multi-class classifications.


## Training and Predicting:
(How to run the code.)
### Setup:
1. In `model/resnet.py`, in line 6, U need to modify the python path:
```
  sys.path.append('{CURRENT_PATH}' + 'models/utils')
```

2. Modify training data path in config files:
- For example, in `inception.cfg`, at line 5 and line 6: `"img_path"` is the figure path (the path of `Images` folder), 
`"label_file"` is the full path of label.csv.

3. Modify testing data path in `pred_main.py`.
- In line 20 and 21:

```
TEST_PATH = "{The path of `Images` folder}" for the testing data
TEST_LABEL_FILE = "{The full path of the sample_submission.csv}"
```

4. Modify labels in the config file:
- In line 4" `"category" = "{expected label name}"`

5. You do not need to worry about the `checker_path` since it using a relative path.


### Run the code:
In `pkugoodspeed` folder, run the following command:

```
python pred_main.py --c configs/xception.cfg
```
Using different config files for different models. (`kerasresnet` is using resnet50 by simply modify the implementation by keras,
 `resnet` is the resnet constructed by myself, but does not work well.)
 
 
### To load and existing model to predict: 
Just put the weights (`.h5`) file in the `output/checkpoints` folder, and modify `config->train_kargs->epochs` into `0`, then run the above command.


### In addition:
- `main.py` is used for testing model accuracy and adjusting hyper parameters.
- `oof_main.py` uses 5 fold to train and generate OOF (out of fold) predictions. (This one is very time consuming, so we did not use it.)
