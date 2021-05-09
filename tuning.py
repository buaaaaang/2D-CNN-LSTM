from keras.models import Sequential
from keras import layers
from keras import optimizers
import kerastuner as kt
import IPython
import h5py
import tensorflow as tf
import numpy as np
from numpy import newaxis

def load_data():
    file = "C:/Users/LG/Desktop/skt/KoreanSpeechDataForSER/2D.hdf5"
    hf = h5py.File(file,'r')
    '''
    x_tr = (hf['x'][:16000])[:,:,:,newaxis]
    x_v = (hf['x'][16000:20000])[:,:,:,newaxis]
    x_t = (hf['x'][20000:])[:,:,:,newaxis]

    y = hf['y']
    y_tr = y[:16000]
    y_v = y[16000:20000]
    y_t = y[20000:]
    '''

    x_tr = (hf['x'][:16])[:,:,:,newaxis]
    x_v = (hf['x'][16:20])[:,:,:,newaxis]
    x_t = (hf['x'][20:30])[:,:,:,newaxis]

    y = hf['y']
    y_tr = y[:16]
    y_v = y[16:20]
    y_t = y[20:30]

    return x_tr, y_tr, x_v, y_v, x_t, y_t

def model(hp):
    shape = (128,251,1)
    unit = 32

    model = Sequential(name='2D')

    model.add(layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',input_shape = shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.MaxPooling2D(pool_size=(4,4), strides=(4,4)))

    model.add(layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.MaxPooling2D(pool_size=(4,4), strides=(4,4)))

    model.add(layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.MaxPooling2D(pool_size=(4,4), strides=(4,4)))

    model.add(layers.Reshape((-1,128)))

    model.add(layers.LSTM(units=unit))

    model.add(layers.Dense(units=5,activation='softmax'))
    
    opt = optimizers.Adam(learning_rate=hp.Float('learning_rate',1e-7,1e-2,sampling='log'))
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    return model

class Tuner(kt.tuners.Hyperband): 
    def run_trial(self,trial,*args,**kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size',16,256,step=16)
        super(Tuner,self).run_trial(trial,*args,**kwargs)

if __name__ == "__main__":
    x_tr, y_tr, x_v, y_v, x_t, y_t = load_data()
    tuner = Tuner(model,objective='val_categorical_accuracy',max_epochs=100,
                  project_name='hello')
    tuner.search(x_tr,y_tr,epochs=100,validation_data=(x_v,y_v),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps['learning_rate'])
    print(best_hps['batch_size'])
    