import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras import models
from models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Dropout
from tensorflow import keras
from keras import optimizers 
from optimizers import Adam
import matplotlib.pyplot as plt

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
    model.add(Conv2D(8,(4,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 1)))

    #First temporal
    model.add(Conv2D(32, (1, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    #Second temporial
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
    loss_tab = []
    accuracy_tab = []


    for epoch in range(CNN_EPOCHS):

        for batch_start in range(0, len(X_train), batch_size):
            batch_end = batch_start + batch_size
            x_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            model.train_on_batch(x_batch, y_batch)
        loss , accuracy = model.evaluate(X_train, y_train, verbose=0)
        loss_tab.append(loss)
        accuracy_tab.append(accuracy)

        print(f"Epoka: {epoch + 1} Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    #model.fit(X_train, y_train, epochs=CNN_EPOCHS, validation_data=(X_test, y_test))

    _, final_accuracy = model.evaluate(X_test, y_test, verbose=2)

    model.save(f"{CNN_MODELS_PATH}/{round(final_accuracy, 4)}.h5")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(CNN_EPOCHS, loss_tab, label='Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(CNN_EPOCHS, accuracy_tab, label='Training Accuracy')

    plt.show()




    return round(final_accuracy, 4)


'''
WNIOSKI:

Budując model od dwóch warstw konwolucyjnych po cztery zauważyłem następujące zależności:
    - Małe modele produkują niski loss i accuracy
    - Zbyt niskie parametry zmniejszają loss i accuracy
    - Zbyt wysokie parametry znacznie bardziej zwiększają loss niż accuracy
    - Zwiększanie modelu o kolejne warstwy zwiększa loss i accuracy
    - Zbyt duży model zwiększa loss i zmniejsza accuracy
    - Odpowiednie dobranie parametrów redukuje loss i zwiększa accuracy

OPTYMALNE USTAWIENIA: [loss: 0.6518 - accuracy: 0.8175]
        model = Sequential()

        model.add(Conv2D(12, (12, 1), input_shape=CNN_INPUT_SHAPE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2, 1)))

        model.add(Conv2D(32, (8, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2, 1)))

        model.add(Conv2D(128, (1, 128), activation='relu'))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(1, 1)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
'''