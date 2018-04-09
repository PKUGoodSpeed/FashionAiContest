from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Dropout, Lambda, Dense
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate, BatchNormalization
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

class DenseNet:
    input_shape = None
    output_dim = None
    model = None
    
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
    
    def buildModel(self, kargs_list, dense_list):
        '''
        kargs_list: list of key arguments
        For dense layers, the dropouts are always 0.5
        den_list: key arg list for den layers
        '''
        in_layer = Input(self.input_shape)
        
        # build dense blocks using the kargs shown in the kargs list
        layerA = in_layer
        for kargs in kargs_list:
            layerB = BatchNormalization(axis=-1) (Conv2D(**kargs) (layerA))
            comb1 = concatenate([layerA, layerB])
            layerC = BatchNormalization(axis=-1) (Conv2D(**kargs) (comb1))
            comb2 = concatenate([comb1, layerC])
            layerA = BatchNormalization(axis=-1) (MaxPooling2D((2, 2)) (Conv2D(**kargs) (comb2)))
        
        # denlayer = GlobalAveragePooling2D() (layerA)
        denlayer = GlobalMaxPooling2D() (layerA)
        
        # adding dense layers
        for kargs in dense_list:
            denlayer = Dropout(0.36) (Dense(**kargs) (denlayer))
        
        out_layer = Dense(self.output_dim, activation='softmax') (denlayer)
        self.model = Model(inputs=[in_layer], outputs=[out_layer])
    
    def getModel(self):
        self.model.summary()
        return self.model