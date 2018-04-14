import os
import sys
import numpy as np
import pandas as pd
# Using K-fold for stacking
from sklearn.cross_validation import KFold

sys.path.append('../utils')
sys.path.append('./models')
from opts_parser import getopts
from process import ImagePrec
from train import Trainer
from dennet import DenseNet
from naive import NaiveCnnNet, VggNet, Vgg19Net
from resnet import ResNet, KerasResNet
from xception import XceptionNet
from inception import InceptionNet

TEST_PATH = '../data/rank'
TEST_LABEL_FILE = '../data/rank/Tests/answer_mock.csv'

model_dict = {
    'densenet': DenseNet,
    'naive': NaiveCnnNet,
    'vgg': VggNet,
    'resnet': ResNet,
    'kerasresnet': KerasResNet,
    'vgg19': Vgg19Net,
    'xception': XceptionNet,
    'inception': InceptionNet
}

def _encode(ans):
    """ encode the prediction into submission format """
    return ";".join(["{0:.4f}".format(x) for x in ans])

if __name__ == '__main__':
    # getting configs
    C = getopts()
    print C
    
    ip = ImagePrec(**C['proc'])
    df = ip.getDataFrame()
    oof_pred = ["" for i in range(len(df))]
    ishape = ip.getInputShape()
    odim = ip.getOutputDim()
    # get model instance
    model_obj = model_dict[C['model_name']](input_shape=ishape, output_dim=odim)

## Creating OOF
"""
    kf = KFold(len(df), n_folds=5, random_state=17)
    for i, (train_index, valid_index) in enumerate(kf):
        model_obj.buildModel(**C['model_kargs'])
        print("Starting Fold # {0} OOF ...".format(str(i)))
        x, y = ip.getbatch(idx=train_index, **C['batch'])
        print("Input shape:")
        print(x.shape)
        print("\nOutput shape:")
        print(y.shape)
        valid_x, valid_y = ip.getbatch(idx=valid_index)
        model = model_obj.getModel()
        trainer = Trainer(model=model, model_name=C['model_name'])
        res = trainer.train(x, y, valid_set=(valid_x, valid_y), **C['train_args'])
        fold_prediction = trainer.predict(valid_x, checker_path=C['train_args']['checker_path'])
        for j, ans in zip(valid_index, fold_prediction):
            oof_pred[j] = _encode(ans)

    df['oof_pred'] = oof_pred
    oof_path = './output/' + C['model_name']
    if not os.path.exists(oof_path):
        os.makedirs(oof_path)
    oof_file = oof_path + "/{LAB}_oof.csv".format(LAB=C["proc"]["category"])
    df.to_csv(oof_file, index=False)
"""
    # Training all
    n = len(df)
    n_train = int(n*0.92)
    all_index = np.random.permutation([i for i in range(n)])
    train_index = all_index[: n_train]
    valid_index = all_index[n_train: ]
    x, y = ip.getbatch(idx=train_index, **C['batch'])
    valid_x, valid_y = ip.getbatch(idx=valid_index)
    print("Input shape:")
    print(x.shape)
    print("\nOutput shape:")
    print(y.shape)

    model_obj.buildModel(**C['model_kargs'])
    model = model_obj.getModel()
    trainer = Trainer(model=model, model_name=C['model_name'])
    trainer.train(x, y, valid_set=(valid_x, valid_y), **C['train_args'])
    
    C["proc"]["img_path"] = TEST_PATH
    C["proc"]["label_file"] = TEST_LABEL_FILE
    ip.getTests(**C["proc"])
    test_x = ip.getTestBatch()
    test_df = ip.getDataFrame()
    test_df['class'] = trainer.predict(test_x, checker_path=C['train_args']['checker_path'])
    
    test_file = oof_path + "/{LAB}_test.csv".format(LAB=C["proc"]["category"])
    test_df.to_csv(test_file, index=False)
    