import pandas as pd
import numpy as np
import tensorflow as tf 

from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Activation, Flatten, Dropout, BatchNormalization

import matplotlib.pyplot as plt

## IMPORT DATA ## 

dataTrain = pd.read_csv("../large_files/fashion-mnist_train.csv")
dataTest = pd.read_csv("../large_files/fashion-mnist_test.csv")
dataTrain = dataTrain.to_numpy()
dataTest = dataTest.to_numpy()
np.random.shuffle(dataTrain)
np.random.shuffle(dataTest)

Xtrain = dataTrain[:,1:].reshape(-1,28,28,1)/255.
Xtest = dataTest[:,1:].reshape(-1,28,28,1)/255.


ytrain = dataTrain[:,0]
ytest = dataTest[:,0]

K = len(ytrain)

ytrain = tf.keras.utils.to_categorical(ytrain, K)
ytest = tf.keras.utils.to_categorical(ytest, K)


## BUILD CNN ##

cnn = Sequential()

## ADD LAYERS TO CNN ##

cnn.add(Conv2D(input_shape=[28, 28, 1],filters=32,
               kernel_size=(3,3)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPool2D())

cnn.add(Conv2D(filters=64,kernel_size=(3,3)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPool2D())

cnn.add(Conv2D(filters=128,kernel_size=(3,3)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPool2D())

cnn.add(Flatten())
cnn.add(Dense(units=300))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.2))
cnn.add(Dense(units=K))
cnn.add(Activation('softmax'))

## COMPILE CNN ##

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
r = cnn.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=15, batch_size=32)
print("Returned:", r)

print(r.history.keys())

## PLOT DATA ##
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()





