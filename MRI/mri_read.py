import pickle
from sklearn.model_selection import train_test_split
import numpy as np

from MRI.config import *

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    return X_train, y_train, X_test, y_test

def concatWithGan(ADHD, CONTROL, ADHD_GENERATED, CONTROL_GENERATED):

    for i in range(len(ADHD)):
        ADHD[i] = np.reshape(ADHD[i], (120, 120,1))

    for i in range(len(CONTROL)):
        CONTROL[i] = np.reshape(CONTROL[i], (120, 120,1))

    return ADHD + ADHD_GENERATED, CONTROL + CONTROL_GENERATED

def makeValidData(ADHD_raw, CONTROL_raw):

    ADHD = []

    ADHD_UPDATED = []

    CONTROL = []

    CONTROL_UPDATED = []

    adhd_Random = np.random.randint(0,len(ADHD_raw), 5)

    control_Random = np.random.randint(0,len(CONTROL_raw), 5)

    for i in range(len(ADHD_raw)):
        if i in adhd_Random:
            ADHD.append(ADHD_raw[i])
        else:
            ADHD_UPDATED.append(ADHD_raw[i])

    for i in range(len(CONTROL_raw)):
        if i in control_Random:
            CONTROL.append(CONTROL_raw[i])
        else:
            CONTROL_UPDATED.append(CONTROL_raw[i])

    y_ADHD = np.ones((len(ADHD)))

    y_CONTROL = np.zeros((len(CONTROL)))

    y_val = np.hstack((y_ADHD, y_CONTROL))

    X_ADHD = np.reshape(ADHD, (len(ADHD), 120, 120, 1))

    X_CONTROL = np.reshape(CONTROL, (len(CONTROL), 120, 120, 1))

    X_val = np.vstack((X_ADHD, X_CONTROL))

    return X_val, y_val, ADHD_UPDATED, CONTROL_UPDATED