import pickle
from sklearn.model_selection import train_test_split
import numpy as np

from config import *

def readPickle(nazwa):
    with open(nazwa, 'rb') as file:
        loaded_data = pickle.load(file)

    return loaded_data

def savePickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def prepareForCnn(ADHD, CONTROL):
    y_ADHD = np.ones((len(ADHD)))

    y_CONTROL = np.zeros((len(CONTROL)))

    y = np.hstack((y_ADHD, y_CONTROL))

    X_ADHD = np.reshape(ADHD,(len(ADHD), 120, 120, 1))

    X_CONTROL = np.reshape(CONTROL, (len(CONTROL), 120, 120, 1))

    X = np.vstack((X_ADHD, X_CONTROL))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=True)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VALIDATE_RATIO, shuffle=True)

    return X_train, y_train, X_test, y_test, X_val, y_val

def concatWithGan(ADHD, CONTROL):
    pass