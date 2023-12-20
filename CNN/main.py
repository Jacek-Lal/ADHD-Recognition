from training import check_saved_trained_models, save_trained_models, get_prepared_model
from eeg_read import getCNNData
from model_config import model
import tensorflow as tf

if __name__ == '__main__':

    if tf.test.is_gpu_available():

        physical_devices = tf.config.list_physical_devices('GPU')
        print("Dostępne karty graficzne:")
        for device in physical_devices:
            print(f"- {device.name}")
    else:
        print("Nie znaleziono dostępnych kart graficznych. Używane będą obliczenia na CPU.")

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
        case _:
            print("Input error")
            exit()
