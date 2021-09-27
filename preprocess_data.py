# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:07:09 2021

@author: Master4
"""
import os
import numpy as np

from json_tricks import dump, load

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr
import pywt

def find_emotion_T(name): 
        if('neutral' in name): return "01"
        elif('happy' in name): return "03"
        elif('sad' in name): return "04"
        elif('angry' in name): return "05"
        elif('fear' in name): return "06"
        elif('disgust' in name): return "07"
        elif('ps' in name): return "08"
        else: return "-1"
        
        
# 'emotions' list fix for classification purposes:
#     Classification values start from 0, Thus an 'n = n-1' operation has been executed for both RAVDESS and TESS databases:
def emotionfix(e_num):
    if e_num == "01":   return 0 # neutral
    elif e_num == "02": return 1 # calm
    elif e_num == "03": return 2 # happy
    elif e_num == "04": return 3 # sad
    elif e_num == "05": return 4 # angry
    elif e_num == "06": return 5 # fear
    elif e_num == "07": return 6 # disgust
    else:  return 7

data_path = './raw_data'

max_len = 0

for subdir, dirs, files in os.walk(data_path):
  for file in files: 
    x, sr = librosa.load(path = os.path.join(subdir,file), sr = None)
    xt, index = librosa.effects.trim(x, top_db=30)
    if len(xt) > max_len:
        max_len = len(xt)

print('Maximum sample length:', max_len)

# Initialize data lists
rms = []
zcr = []
mfcc = []
emotions = []

# Initialize variables
total_length = 173056 # desired frame length for all of the audio samples.
frame_length = 2048
hop_length = 512

folder_path = './raw_data' 

i = 0

for subdir, dirs, files in os.walk(folder_path):
    for file in files: 
        i += 1
        print(i)
        try:
            _, sr = librosa.load(path = os.path.join(subdir,file), sr = None)
            rawsound = AudioSegment.from_file(os.path.join(subdir,file)) 
            normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
            normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
            xt, index = librosa.effects.trim(normal_x, top_db=30)
            padded_x = np.pad(xt, (0, total_length-len(xt)), 'constant')
            final_x = nr.reduce_noise(padded_x, sr)
             
            f1 = librosa.feature.rms(final_x, frame_length=frame_length, hop_length=hop_length) # Energy - Root Mean Square   
            f2 = librosa.feature.zero_crossing_rate(final_x , frame_length=frame_length, hop_length=hop_length, center=True) # ZCR      
            f3 = librosa.feature.mfcc(final_x, sr=sr, n_mfcc=13, hop_length = hop_length) # MFCC
            
            if (find_emotion_T(file) != "-1"): 
                  name = find_emotion_T(file)
            else:                             
                  name = file[6:8]                      
        
         # Filling the data lists  
            rms.append(f1)
            zcr.append(f2)
            mfcc.append(f3)
            emotions.append(emotionfix(name)) 
        except:
            print("Korumpirana datoteka:", file)

f_rms = np.asarray(rms).astype('float32')
f_rms = np.swapaxes(f_rms,1,2)
f_zcr = np.asarray(zcr).astype('float32')
f_zcr = np.swapaxes(f_zcr,1,2)
f_mfccs = np.asarray(mfcc).astype('float32')
f_mfccs = np.swapaxes(f_mfccs,1,2)

print('ZCR shape:',f_zcr.shape)
print('RMS shape:',f_rms.shape)
print('MFCCs shape:',f_mfccs.shape)

X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)

Y = np.asarray(emotions).astype('int8')
Y = np.expand_dims(Y, axis=1)

x_data = X.tolist() 
x_path = './processed_data_json/X_data.json' 
dump(obj = x_data, fp = x_path)

y_data = Y.tolist() 
y_path = './processed_data_json/Y_data.json'
dump(obj = y_data, fp = y_path)
