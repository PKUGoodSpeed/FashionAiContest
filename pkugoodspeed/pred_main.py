import os
import sys
import numpy as np
import pandas as pd
import pickle
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
    # Training all
    n = len(df)
    kf = KFold(n, n_folds=10, random_state=17)
    train_index = None
    valid_index = None
    for t, v in kf:
        print "wtf"
        train_index = t
        valid_index = v
        break
    print len(train_index), len(valid_index)
    x, y = ip.getbatch(idx=train_index, **C['batch'])
    valid_x, valid_y = ip.getbatch(idx=valid_index)
    ip._imgs = None
    print("Input shape:")
    print(x.shape)
    print("\nOutput shape:")
    print(y.shape)

    model_obj.buildModel(**C['model_kargs'])
    model = model_obj.getModel()
    trainer = Trainer(model=model, model_name=C['model_name'])
    trainer.train(x, y, valid_set=(valid_x, valid_y), **C['train_args'], resume=True)
    
    oof_path = './output/' + C['model_name']
    if not os.path.exists(oof_path):
        os.makedirs(oof_path)
    
    oof_fnames = np.array(df['fname'].tolist())[valid_index]
    oof_fnames = list(oof_fnames)
    oof_labels = list(valid_y)
    oof_pred = trainer.predict(valid_x, checker_path=C['train_args']['checker_path'])
    oof_pred = list(oof_pred)
    
    output_df = pd.DataFrame({
        'fname': oof_fnames,
        'label': oof_labels,
        'pred': oof_pred
    })
    
    oof_file = oof_path + "/{LAB}_oof.pik".format(LAB=C["proc"]["category"])
    print("Saving oof in {fname}".format(fname=oof_file))
    pickle.dump(output_df, open(oof_file, 'wb'))

    ## show evaluation
    model = trainer.getModel()
    score = model.evaluate(valid_x, valid_y, batch_size=16, verbose=1)
    print("===============================================")
    print("Evaluation:")
    print score
    print("===============================================")

    C["proc"]["img_path"] = TEST_PATH
    C["proc"]["label_file"] = TEST_LABEL_FILE
    ip.getTests(**C["proc"])
    test_x = ip.getTestBatch()
    test_df = ip.getDataFrame()
    test_df['pred'] = list(trainer.predict(test_x, checker_path=C['train_args']['checker_path']))
    output_df = test_df[['fname', 'pred']]

    
    test_file = oof_path + "/{LAB}_test.pik".format(LAB=C["proc"]["category"])
    print("Saving test_predictions in {fname}".format(fname=oof_file))
    pickle.dump(output_df, open(test_file, 'wb'))
    