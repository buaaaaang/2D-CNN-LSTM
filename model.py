from keras.models import Sequential
from keras import layers
from keras import optimizers



def model():#hp):
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

    
    #LSTM
    model.add(layers.Reshape((-1,128)))
    model.add(layers.LSTM(units=unit))
    model.add(layers.Dense(units=5,activation='softmax'))
    opt = optimizers.Adam(learning_rate = 0.00067)
    

    '''
    #Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(units=88,activation='relu'))
    model.add(layers.Dense(units=120,activation='relu'))
    model.add(layers.Dense(units=K,activation='softmax'))
    opt = optimizers.Adam(learning_rate=0.00026)
    '''

    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    return model


    
    
    
