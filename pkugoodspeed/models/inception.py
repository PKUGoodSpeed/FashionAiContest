from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Dropout, Lambda, Dense, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate, Add
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, merge, Flatten
from keras.applications import InceptionV3

class InceptionNet:
    input_shape = None
    output_dim = None
    model = None
    
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
    
    def buildModel(self, dense_list):
        '''
        n_filters: num of filters in the initial layers
        depth: depth of the resnet
        For dense layers, the dropouts are always 0.5
        den_list: key arg list for den layers
        '''
        in_layer = Input(self.input_shape)
        
        # use keras existing resnet50
        inceptionModel = InceptionV3(include_top=False, weights='imagenet', 
        input_tensor=Input(shape=self.input_shape), pooling="avg")

        kernel = Dropout(0.5) (inceptionModel (in_layer))
        inceptionModel.summary()
        print kernel.shape

        denlayer = kernel
        # denlayer = GlobalMaxPooling2D() (kernel)
        # denlayer = Flatten() (kernel)
        
        # adding dense layers
        for kargs in dense_list:
            denlayer = Dropout(0.5) (Dense(**kargs) (denlayer))
        
        out_layer = Dense(self.output_dim, activation='softmax') (denlayer)
        self.model = Model(inputs=[in_layer], outputs=[out_layer])
    
    def getModel(self):
        self.model.summary()
        return self.model