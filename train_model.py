# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:23:54 2021

@author: Master4
"""

import numpy as np

from json_tricks import load


import keras

x_path = './processed_data_json/X_data.json' 
y_path = './processed_data_json/Y_data.json'

X = load(x_path)
X = np.asarray(X, dtype = 'float32')

Y = load(y_path)
Y = np.asarray(Y, dtype = 'int8')

from sklearn.model_selection import train_test_split
x_train, x_tosplit, y_train, y_tosplit = train_test_split(X, Y, test_size = 0.15, random_state = 1)
x_val, x_test, y_val, y_test = train_test_split(x_tosplit, y_tosplit, test_size = 0.3, random_state = 1)

#one-hot encoding
y_train_class = keras.utils.to_categorical(y_train, 8, dtype = 'int8')
y_val_class = keras.utils.to_categorical(y_val, 8, dtype = 'int8')

print(np.shape(x_train))
print(np.shape(x_val))
print(np.shape(x_test))


from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks

model = Sequential()
model.add(layers.LSTM(64, return_sequences = True, input_shape=(X.shape[1:3])))
model.add(layers.LSTM(64))
model.add(layers.Dense(8, activation = 'softmax'))
print(model.summary())

batch_size = 23

checkpoint_path = './model_weights/best_weights.hdf5'
mcp_save = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                           monitor='val_categorical_accuracy',
                           mode='max')

rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', 
                                    factor=0.1, patience=100)
                             

model.compile(loss='categorical_crossentropy', 
                optimizer='RMSProp', 
                metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train_class, 
                      epochs=200, batch_size = batch_size, 
                      validation_data = (x_val, y_val_class), 
                      callbacks = [mcp_save, rlrop])

model.load_weights(checkpoint_path)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Loss function')
plt.ylabel('Loss value')
plt.xlabel('Epoch')

plt.savefig('./plots/train_val_loss.png')

plt.figure()

plt.plot(history.history['categorical_accuracy'], label='Train acc')
plt.plot(history.history['val_categorical_accuracy'], label='Val acc')
plt.title('Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epoch')

plt.savefig('./plots/train_val_acc.png')

y_test_class = keras.utils.to_categorical(y_test, 8, dtype = 'int8')

loss, acc = model.evaluate(x_test, y_test_class, verbose=2)

y_test_class = np.argmax(y_test_class, axis=1)
predictions = model.predict(x_test)
y_pred_class = np.argmax(predictions, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sb
import pandas as pd

cm=confusion_matrix(y_test_class, y_pred_class)

index = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  
columns = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  
 
cm_df = pd.DataFrame(cm,index,columns)                      
plt.figure(figsize=(12,8))
ax = plt.axes()

sb.heatmap(cm_df, ax = ax, fmt="d", annot=True)
ax.set_ylabel('True class')
ax.set_xlabel('Predicted class')

plt.savefig('plots/confusion_matrix.png')

model_json = model.to_json()

with open('./model_json/model.json', "w") as json_file:
    json_file.write(model_json)



