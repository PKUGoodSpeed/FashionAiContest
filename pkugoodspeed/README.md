## Usage

#### Setup:

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


#### Run the code:

In `pkugoodspeed` folder, run the following command:

```
python pred_main.py --c configs/xception.cfg
```
Using different config files for different models. (`kerasresnet` is using resnet50 by simply modify the implementation by keras,
 `resnet` is the resnet constructed by myself, but does not work well.)
 
 
 
 ### Current Resuls:
 
 | models | resnet50 | inception | xception
