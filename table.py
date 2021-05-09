from pandas import DataFrame
from keras.models import load_model
from numpy import newaxis
import h5py
import numpy as np
import tensorflow as tf

#x axis is real one, y axis is prediction
emotion = ["Angry","Disgust","Fear","Neutral","Sadness"] 
data = {'Angry': [0, 0, 0, 0, 0],
    'Disgust': [0, 0, 0, 0, 0],
    'Fear': [0, 0, 0, 0, 0],
    'Neutral': [0, 0, 0, 0, 0],
    'Sadness': [0, 0, 0, 0, 0]}

model = load_model('model.h5')

file = "C:/Users/LG/Desktop/skt/KoreanSpeechDataForSER/2D.hdf5"
hf = h5py.File(file,'r')
x_t = (hf['x'][2000:])[:,:,:,newaxis]
y_t = hf['y'][2000:]

for i in range(x_t.shape[0]):
    x = x_t[i]
    y = y_t[i]
    y_p = model(np.array([x]))
    real = np.argmax(y)
    prediction = tf.math.argmax(y_p[0]).numpy()
    data[emotion[real]][prediction] += 1




dataframe = DataFrame(data,index=emotion)
print(dataframe,flush=True)