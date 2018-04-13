import os
from keras.optimizers import SGD, Adam, Adadelta, RMSprop, Adagrad
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ModelCheckpoint

global_learning_rate = 0.01
global_decaying_rate = 0.92

class Trainer:
    model = None
    model_name = None
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
    
    def train(self, x, y, valid_set=None, learning_rate=0.02, decaying_rate=0.9, epochs=10, checker_path='../output/checkpoints'):
        '''train the model'''
        # compile the model first
        self.model.compile(optimizer=Adam(0.005), loss='categorical_crossentropy', metrics=['accuracy'])
        
        global global_learning_rate
        global global_decaying_rate
        ## Setting learning rate explicitly
        global_learning_rate = learning_rate
        global_decaying_rate = decaying_rate
        
        ## Adaptive learning rate changing
        def scheduler(epoch):
            global global_learning_rate
            global global_decaying_rate
            if epoch%4 == 0:
                global_learning_rate *= global_decaying_rate
                print("CURRENT LEARNING RATE = " + str(global_learning_rate))
            return global_learning_rate
        change_lr = LearningRateScheduler(scheduler)
        
        ## Set early stopper:
        earlystopper = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
        
        ## Set Check point
        if not os.path.exists(checker_path):
            os.makedirs(checker_path)
        checker = "{PATH}/{MODEL}.h5".format(PATH=checker_path, MODEL=self.model_name)
        checkpointer = ModelCheckpoint(filepath=checker, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        
        if valid_set is None:
            history = self.model.fit(x, y, epochs=epochs, verbose=1, validation_split=0.05, batch_size=16,
            callbacks=[earlystopper, checkpointer, change_lr])
        else:
            history = self.model.fit(x, y, epochs=epochs, verbose=1, validation_data=valid_set, batch_size=16,
            callbacks=[earlystopper, checkpointer, change_lr])
        return history
    
    def save(self, path):
        ''' saving the model '''
        if not os.path.exists(path):
            os.makedirs(path)
        model_file = path + '/' + self.model_name
        print("Saving the model into {FILE} ...".format(FILE=model_file))
        self.model.save(model_file)
    
    def load(self, model_file):
        ''' loading the model '''
        print("Loading model from {FILE} ...".format(FILE=model_file))
        self.model.load_weights(model_file)
    
    def predict(self, test_x):
        ''' make predictions '''
        print(" Making predictions ...")
        return self.model.predict(test_x)
