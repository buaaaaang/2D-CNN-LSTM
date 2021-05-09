import csv
import wave
from pathlib import Path
import time
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
import librosa

random.seed(20210507)
path = "C:/Users/LG/Desktop/skt/KoreanSpeechDataForSER"
emotion = ["Angry","Disgust","Fear","Neutral","Sadness"] 
#we will noet consider happiness and surprise (there are only few samples of those emotions)
len_audio = 8
sr = 48000
n_total = 645*len(emotion)

def matchsize(x):
    if len(x) < sr*len_audio:
        pad = sr*len_audio-len(x)
        #x = np.pad(x,(pad//2,pad-pad//2),'constant',constant_values=0)
        x = np.pad(x,(0,pad),'constant',constant_values=0)                
    else: 
        x = x[:sr*len_audio]
    return x

def log_mel(y):
    y = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=len_audio*sr*4//250,hop_length=len_audio*sr//250)
    return librosa.power_to_db(y, ref=np.max)

def preprocess(filepath):
    with h5py.File(filepath,'w') as h:
        num = {"Angry":0,"Disgust":0,"Fear":0,"Neutral":0,"Sadness":0}
        x_total = h.create_dataset('x',(n_total,128,251),dtype='float32')
        y_total = h.create_dataset('y',(n_total,len(emotion)),dtype='float32')
        
        index = np.array(range(len(x_total)))
        np.random.shuffle(index)

        f = open(path+'/4th.csv','r')
        rdr = csv.reader(f)
        next(rdr)
        i = 0
        for line in rdr:
            if (line[3] in emotion) and (num[line[3]]<645):
                x, _ = librosa.load(path+"/4th/"+line[0]+".wav", sr=sr)
                x = matchsize(x)
                x = log_mel(x)
                x_total[index[i]] = (np.float32(x))
                y = np.zeros(len(emotion))
                y[emotion.index(line[3])] = 1.0
                y_total[index[i]] = y
                num[line[3]] += 1
                if (i%1000==0): print(i,flush=True)
                i += 1
        f.close()

        f = open(path+'/5th.csv','r')
        rdr = csv.reader(f)
        next(rdr)
        for line in rdr:
            if line[3] in emotion and (num[line[3]]<645):
                x, _ = librosa.load(path+"/5th/"+line[0]+".wav", sr=sr)
                x = matchsize(x)
                x = log_mel(x)
                x_total[index[i]] = (np.float32(x))
                y = np.zeros(len(emotion))
                y[emotion.index(line[3])] = 1.0
                y_total[index[i]] = y
                num[line[3]] += 1
                if (i%1000==0): print(i,flush=True)
                i += 1
        f.close()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    import librosa.display
    #checking function log-mel
    '''
    f = open(path+'/4th.csv','r')
    rdr = csv.reader(f)
    next(rdr)
    no = 7067
    for i in range(no): line = next(rdr)
    x, _ = librosa.load(path+"/4th/"+line[0]+".wav", sr=sr)
    x = matchsize(x)
    x = log_mel(x)
    print(x.shape)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(x, x_axis='time',y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img,ax=ax,format='%+2.0f dB')
    plt.savefig('log_mel_sample.png')
    plt.show()
    '''
    
    #preprocess
    preprocess(path + "/2D.hdf5")
    file = path + '/2D.hdf5'
    hf = h5py.File(file,'r')

    print(len(hf['x']))
    print(hf['x'][0].shape)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(hf['x'][0], x_axis='time',y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img,ax=ax,format='%+2.0f dB')
    plt.show()
    



