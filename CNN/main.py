import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from training import *
from eeg_read import *
from model_config import *
import tensorflow as tf
from one_patient_predict import *

devices = tf.config.list_physical_devices()

gpu_devices = [device for device in devices if 'GPU' in device.device_type]

if len(gpu_devices) > 0:
    print("TensorFlow używa karty graficznej do obliczeń.")
else:
    print("TensorFlow używa CPU lub nie ma dostępnej karty graficznej.")

print("---------------------------------------------------------------------------------------")

cnn = None

user_choice = input("Do you want to train a new model (enter 'train') or load an existing one (enter 'load')? ").lower()

match user_choice:
    case "train":
        x_train, x_test, y_train, y_test = getCNNData()
        cnn, cnn_accuracy = get_prepared_model(model, x_train, y_train, x_test, y_test)
        save_trained_models(cnn, cnn_accuracy)
    case "load":
        trained_model = check_saved_trained_models()

        if trained_model is None:
            print("No existing trained model found. Please train a new model.")
            exit()
        
        cnn = trained_model

        result, chance = predictPatient(cnn, PATIENT_INPUT_FILE)

        print("Chance of ADHD: ", chance)
        if result:
            print("Patient has ADHD")
        else:
            print("Patient doesn't have ADHD")

    case _: 
        print("Input error")
        exit()
