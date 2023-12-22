from tensorflow import keras
from keras import layers
from config import CNN_INPUT_SHAPE

model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=CNN_INPUT_SHAPE, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')
    ])
