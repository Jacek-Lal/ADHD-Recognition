import os
import re
from tensorflow import keras
from config import CNN_MODELS_PATH, CNN_EPOCHS


def check_saved_trained_models():
    trained_model_files = [f for f in os.listdir(CNN_MODELS_PATH) if f.endswith('.h5')]
    if trained_model_files:
        # Wydobywanie precyzji z nazw plików i wybieranie największej
        max_accuracy = max([float(re.search(r"(\d+\.\d+).h5", file).group(1)) for file in trained_model_files])
        trained_model_path = os.path.join(CNN_MODELS_PATH, f'{max_accuracy:.4f}.h5')
        trained_model = keras.models.load_model(trained_model_path)
        print(f"Trained model loaded from file: {trained_model_path}")
        return trained_model
    else:
        return None


def save_trained_models(trained_model, final_accuracy):
    if not os.path.exists(CNN_MODELS_PATH):
        os.makedirs(CNN_MODELS_PATH)
    trained_model_path = os.path.join(CNN_MODELS_PATH, f'{final_accuracy:.4f}.h5')
    trained_model.save(trained_model_path)
    print(f"Trained model saved to file: {trained_model_path}")


def get_prepared_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=CNN_EPOCHS, batch_size=2)
    _, final_accuracy = model.evaluate(x_test, y_test)
    print(f"Final accuracy: {final_accuracy}")

    return model, final_accuracy
