import os
import sys
import numpy as np
import pandas as pd

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

if __name__ == '__main__':
    # getting configs
    C = getopts()
    print C
    
    ip = ImagePrec(**C['proc'])
    idx = [i for i in range(len(ip._labels))]
    x, y = ip.getbatch(idx=idx, **C['batch'])
    print("Input shape:")
    print(x.shape)
    print("\nOutput shape:")
    print(y.shape)
    
    # get model instance
    model_obj = model_dict[C['model_name']](input_shape=x.shape[1:], output_dim=y.shape[1])
    model_obj.buildModel(**C['model_kargs'])
    model = model_obj.getModel()
    
    # get trainer
    trainer = Trainer(model=model, model_name=C['model_name'])
    trainer.train(x, y, valid_set=None, **C['train_args'])
    