import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Dropout

import sys

# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from EEG.config import *

def CnnFit(X_train, y_train, X_test, y_test):

    model = Sequential()

    # First spatial
    model.add(Conv2D(16, (10, 1), input_shape=CNN_INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 1)))

    # Second spatial
    model.add(Conv2D(8, (4, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 1)))

    # First temporal
    model.add(Conv2D(32, (1, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    # Second temporial
    model.add(Conv2D(16, (1, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    #Feature selection
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    optimizer = Adam(learning_rate=0.0001)

    # Compile
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    batch_size = 32


    for epoch in range(CNN_EPOCHS):

        for batch_start in range(0, len(X_train), batch_size):
            batch_end = batch_start + batch_size
            x_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            model.train_on_batch(x_batch, y_batch)
        loss , accuracy = model.evaluate(X_train, y_train, verbose=0)

        print(f"Epoch: {epoch + 1} Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    #model.fit(X_train, y_train, epochs=CNN_EPOCHS, validation_data=(X_test, y_test))

    _, final_accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.save(f'../MODEL/{round(final_accuracy, 4)}.h5')

    return round(final_accuracy, 4)