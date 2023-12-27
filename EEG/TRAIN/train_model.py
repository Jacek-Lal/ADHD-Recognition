import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, AveragePooling2D

import sys

# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import *

def CnnFit(X_train, y_train, X_test, y_test):   #funkcja z artykułu

    model = Sequential()

    #First spatial
    model.add(Conv2D(16, (10, 1), input_shape=CNN_INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 1)))

    #Second spatial
    model.add(Conv2D(16,(4,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 1)))

    #First temporal
    model.add(Conv2D(32, (1, 32), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 16)))

    # #Second temporial
    # model.add(Conv2D(32, (1, 16), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(AveragePooling2D(pool_size=(1, 8)))

    #Feature selection
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=CNN_EPOCHS, validation_data=(X_test, y_test))

    _, final_accuracy = model.evaluate(X_test,  y_test, verbose=2)

    model.save(f"{CNN_MODELS_PATH}/{round(final_accuracy, 4)}.h5")

    return round(final_accuracy, 4)

def CnnFit_test(X_train, y_train, X_test, y_test):

    model = Sequential()

    #First spatial
    model.add(Conv2D(16, (3, 3), input_shape=CNN_INPUT_SHAPE, activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    #Second spatial
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    # #First temporal
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(AveragePooling2D(pool_size=(1, 2)))
    #
    # #Second temporial
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(AveragePooling2D(pool_size=(1, 2)))

    #Feature selection
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=CNN_EPOCHS, validation_data=(X_test, y_test))

    _, final_accuracy = model.evaluate(X_test,  y_test, verbose=2)

    model.save(f"{CNN_MODELS_PATH}/{round(final_accuracy, 4)}.h5")

    return round(final_accuracy, 4)
