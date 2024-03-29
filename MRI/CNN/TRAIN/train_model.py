from keras.optimizers import Adam
from keras import models, layers

from MRI.config import *

def CnnFit(X_train, y_train, X_test, y_test, save, model_path):

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=CNN_INPUT_SHAPE_MRI))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    for epoch in range(CNN_EPOCHS_MRI):

        for batch_start in range(0, len(X_train), BATCH_SIZE_MRI):
            batch_end = batch_start + BATCH_SIZE_MRI
            x_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            model.train_on_batch(x_batch, y_batch)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=0)

        print(f"Epoch: {epoch + 1} Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    _, final_accuracy = model.evaluate(X_test, y_test, verbose=0)

    if save == True:
        model.save(rf'{model_path}/{round(final_accuracy, 4)}.h5')

    return round(final_accuracy, 4)