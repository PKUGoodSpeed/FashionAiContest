from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Dropout, Lambda, Dense, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate, Add
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, merge, Flatten

class ResNet:
    input_shape = None
    output_dim = None
    model = None
    
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
    
    def buildModel(self, n_filters, depth, dense_list):
        '''
        n_filters: num of filters in the initial layers
        depth: depth of the resnet
        For dense layers, the dropouts are always 0.5
        den_list: key arg list for den layers
        '''
        in_layer = Input(self.input_shape)
        
        # build resnet blocks
        n_filters = n_filters
        dropout_ratio = 0.0
        kernel = BatchNormalization(axis=-1) (Conv2D(filters=n_filters, kernel_size=3, padding="same") (in_layer))
        kernel = Activation('relu') (kernel)
        for _ in range(depth):
            layerA = Conv2D(filters=2*n_filters, kernel_size=3, padding="same") (kernel)
            # layerA = Dropout(dropout_ratio) (layerA)
            layerB = Conv2D(filters=int(n_filters/2), kernel_size=3, padding="same") (kernel)
            layerB = Activation('relu') (BatchNormalization(axis=-1) (layerB))
            layerB = Conv2D(filters=int(n_filters/2), kernel_size=3, padding="same") (layerB)
            layerB = Activation('relu') (BatchNormalization(axis=-1) (layerB))
            layerB = Conv2D(filters=2*n_filters, kernel_size=3, padding="same") (layerB)
            # layerB = Dropout(dropout_ratio) (layerB)
            kernel = merge([layerA, layerB], mode='sum')
            
            for _ in range(3):
                layerB = Activation('relu') (BatchNormalization(axis=-1) (kernel))
                layerB = Conv2D(filters=int(n_filters/2), kernel_size=3, padding="same") (layerB)
                layerB = Activation('relu') (BatchNormalization(axis=-1) (layerB))
                layerB = Conv2D(filters=int(n_filters/2), kernel_size=3, padding="same") (layerB)
                layerB = Activation('relu') (BatchNormalization(axis=-1) (layerB))
                layerB = Conv2D(filters=2*n_filters, kernel_size=3, padding="same") (layerB)
                # layerB = Dropout(min(0.5, 2 * dropout_ratio)) (layerB)
                kernel = merge([kernel, layerB], mode='sum')
            
            kernel = Activation('relu') (BatchNormalization(axis=-1) (kernel))
            kernel = MaxPooling2D((2, 2)) (kernel)
            n_filters *= 2
            dropout_ratio = min(0.4, 2*dropout_ratio)
        
        # denlayer = GlobalAveragePooling2D() (kernel)
        # denlayer = GlobalMaxPooling2D() (layerA)
        denlayer = Flatten() (kernel)
        
        # adding dense layers
        for kargs in dense_list:
            denlayer = Dropout(0.6) (Dense(**kargs) (denlayer))
        
        out_layer = Dense(self.output_dim, activation='softmax') (denlayer)
        self.model = Model(inputs=[in_layer], outputs=[out_layer])
    
    def getModel(self):
        self.model.summary()
        return self.model
