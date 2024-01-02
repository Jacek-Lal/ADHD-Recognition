import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Dropout

from config import *

def CnnFit_test(X_train, y_train, X_test, y_test):

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

    # Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=CNN_EPOCHS, validation_data=(X_test, y_test))

    _, final_accuracy = model.evaluate(X_test, y_test, verbose=2)

    model.save(f"{CNN_MODELS_PATH}/{round(final_accuracy, 4)}.h5")

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