import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, AveragePooling2D
from keras import layers
from tensorflow import keras


def CnnFit1(X_train, y_train, X_test, y_test, frameSize, numEpochs):
    model = Sequential()

    #First spatial
    model.add(Conv2D(16, (10, 1), input_shape=(19, frameSize, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 1)))

    #Second spatial
    model.add(Conv2D(16,(4,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 1)))

    #First temporal
    model.add(Conv2D(32, (1, 128), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 64)))

    #Second temporial
    model.add(Conv2D(32, (1, 64), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 32)))

    #Feature selection
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    #Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    model.fit(X_train, y_train, epochs=numEpochs, validation_data=(X_test, y_test))

    _, final_accuracy = model.evaluate(X_test,  y_test, verbose=2)

    return model, final_accuracy

def CnnFit1(X_train, y_train, X_test, y_test, frameSize, numEpochs):
    model = Sequential()

    #First spatial
    model.add(Conv2D(16, (3, 3), input_shape=(19, frameSize, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    #Second spatial
    model.add(Conv2D(16,(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    #First temporal
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    #Second temporial
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    #Feature selection
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    #Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=numEpochs, validation_data=(X_test, y_test))

    _, final_accuracy = model.evaluate(X_test,  y_test, verbose=2)

    return model, final_accuracy
