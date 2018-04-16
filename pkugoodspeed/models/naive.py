from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Dropout, Lambda, Dense
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate, BatchNormalization
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.applications import VGG16, VGG19

class NaiveCnnNet:
    input_shape = None
    output_dim = None
    model = None
    
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
    
    def buildModel(self, kargs_list, dropout_list, dense_list):
        '''
        kargs_list: list of key arguments
        dropout_list: list of dropout ratios
        For dense layers, the dropouts are always 0.5
        den_list: key arg list for den layers
        '''
        in_layer = Input(self.input_shape)
        assert len(kargs_list) == len(dropout_list), "key arguments and dropouts do not match."
        
        # build stacked cnn layers
        tmp = in_layer
        for kargs, doratio in zip(kargs_list, dropout_list):
            tmp = MaxPooling2D((2, 2)) (Dropout(doratio) (Conv2D(**kargs) (tmp)))
        
        # denlayer = GlobalAveragePooling2D() (layerA)
        # denlayer = GlobalMaxPooling2D() (layerA)
        denlayer = Flatten() (tmp)
        
        # adding dense layers
        for kargs in dense_list:
            denlayer = Dropout(0.5) (Dense(**kargs) (denlayer))
        
        out_layer = Dense(self.output_dim, activation='softmax') (denlayer)
        self.model = Model(inputs=[in_layer], outputs=[out_layer])
    
    def getModel(self):
        self.model.summary()
        return self.model

class VggNet:
    input_shape = None
    output_dim = None
    model = None
    
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
    
    def buildModel(self, dense_list):
        '''
        For dense layers, the dropouts are always 0.5
        den_list: key arg list for den layers
        '''
        in_layer = Input(self.input_shape)
        
        # build vgg_block
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        vgg16_block = model_vgg16_conv(in_layer)
        model_vgg16_conv.summary()
        vgg16_block = Conv2D(512, kernel_size=2) (vgg16_block)
        vgg16_block = Activation('elu') (BatchNormalization(axis=-1) (vgg16_block))
        vgg16_block = Dropout(0.25) (vgg16_block)
        print vgg16_block.shape
        
        denlayer = GlobalAveragePooling2D() (vgg16_block)
        # denlayer = GlobalMaxPooling2D() (vgg16_block)
        # denlayer = Flatten() (vgg16_block)
        
        # adding dense layers
        for kargs in dense_list:
            denlayer = Dropout(0.66) (Dense(**kargs) (denlayer))
        
        out_layer = Dense(self.output_dim, activation='softmax') (denlayer)
        self.model = Model(inputs=[in_layer], outputs=[out_layer])
    
    def getModel(self):
        self.model.summary()
        return self.model
        
class Vgg19Net:
    input_shape = None
    output_dim = None
    model = None
    
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
    
    def buildModel(self, dense_list):
        '''
        For dense layers, the dropouts are always 0.5
        den_list: key arg list for den layers
        '''
        in_layer = Input(self.input_shape)
        
        # build vgg_block
        model_vgg19_conv = VGG19(weights='imagenet', include_top=False)
        vgg19_block = model_vgg19_conv(in_layer)
        model_vgg19_conv.summary()
        vgg19_block = Conv2D(512, kernel_size=2) (vgg19_block)
        vgg19_block = Activation('elu') (BatchNormalization(axis=-1) (vgg19_block))
        vgg19_block = Dropout(0.32) (vgg19_block)
        print vgg19_block.shape
        
        denlayer = GlobalAveragePooling2D() (vgg19_block)
        # denlayer = GlobalMaxPooling2D() (vgg19_block)
        # denlayer = Flatten() (vgg19_block)
        
        # adding dense layers
        for kargs in dense_list:
            denlayer = Dropout(0.68) (Dense(**kargs) (denlayer))
        
        out_layer = Dense(self.output_dim, activation='softmax') (denlayer)
        self.model = Model(inputs=[in_layer], outputs=[out_layer])
    
    def getModel(self):
        self.model.summary()
        return self.model