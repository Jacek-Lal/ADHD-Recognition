from training import check_saved_trained_models, save_trained_models, get_prepared_model
from eeg_read import getCNNData
from model_config import model
import tensorflow as tf

devices = tf.config.list_physical_devices()

gpu_devices = [device for device in devices if 'GPU' in device.device_type]

if len(gpu_devices) > 0:
    print("TensorFlow używa karty graficznej do obliczeń.")
else:
    print("TensorFlow używa CPU lub nie ma dostępnej karty graficznej.")

print("---------------------------------------------------------------------------------------")

print("pisze bo jacek prosi")
print("pisze bo jacek prosi")
print("pisze bo jacek prosi")

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
    case _: 
        print("Input error")
        exit()
