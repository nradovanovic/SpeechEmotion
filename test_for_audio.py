# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:12:04 2021

@author: Master4
"""

import numpy as np

from json_tricks import dump, load

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr
import pywt

total_length = 173056 
frame_length = 2048
hop_length = 512



_, sr = librosa.load(path = './test_audios/output11.wav', sr = None)
 
rawsound = AudioSegment.from_file('./test_audios/output11.wav') 
 
normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
  
normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
 
final_x = nr.reduce_noise(normal_x, sr)
 
 
f1 = librosa.feature.rms(final_x, frame_length=frame_length, hop_length=hop_length) # Energy - Root Mean Square   
f2 = librosa.feature.zero_crossing_rate(final_x , frame_length=frame_length, hop_length=hop_length, center=True) # ZCR      
f3 = librosa.feature.mfcc(final_x, sr=sr, n_mfcc=13, hop_length = hop_length) # MFCC

X = np.concatenate((f1, f2, f3), axis = 0)

X = np.transpose(X)

X = X.reshape(-1, X.shape[0], X.shape[1])
    
from keras.models import model_from_json

with open('./model_json/model.json' , 'r') as json_file:
    json_savedModel = json_file.read()
    
model = model_from_json(json_savedModel)
model.load_weights('./model_weights/best_weights.hdf5')

model.compile(loss='categorical_crossentropy', 
                optimizer='RMSProp', 
                metrics=['categorical_accuracy'])

klase = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  
import matplotlib.pyplot as plt

plt.bar(klase, model.predict(X).flatten().tolist())
print("Predikcija: " + klase[np.argmax(model.predict(X))])
