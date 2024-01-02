import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from PREDICT.eeg_read import *
from PREDICT.eeg_filter import *
from TRAIN.train_model import *
from config import *
import tkinter as tk
from tkinter import ttk
from tkinter import Tk, Label, Button, StringVar, OptionMenu, Entry, Text, Scrollbar, END, filedialog

# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import *

def predict():
    patient_dir = patient_dir_var.get()
    model_name = model_name_var.get()

    DATA = readEEGRaw(patient_dir)
    DATA_FILTERED = filterEEGData(DATA)
    DATA_CLIPPED = clipEEGData(DATA_FILTERED)
    DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)
    DATA_FRAMED = frameDATA(DATA_NORMALIZED)

    model = load_model(f'{CNN_MODELS_PATH}/{model_name}')
    predictions = model.predict(DATA_FRAMED)

    probability, textStatus = checkResult(predictions)
    result_text.config(state='normal')
    result_text.delete(1.0, END)
    result_text.insert(END, f'Probability: {probability}\nText Status: {textStatus}')
    result_text.config(state='disabled')

def browse_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select EEG Data File")
    patient_dir_var.set(filename)

# GUI setup
root = Tk()
root.title("EEG Prediction GUI")

# Model Dropdown
model_name_var = StringVar(root)
model_name_var.set("[Select Model]")  # default value
models_list = os.listdir("./MODEL")
model_dropdown = OptionMenu(root, model_name_var, *models_list)
model_dropdown_label = Label(root, text="Select Model:")
model_dropdown_label.pack()
model_dropdown.pack()

# File Input
patient_dir_var = StringVar(root)
patient_dir_entry = Entry(root, textvariable=patient_dir_var)
patient_dir_label = Label(root, text="Enter Patient Directory:")
patient_dir_label.pack()
patient_dir_entry.pack()

# Browse Button
browse_button = Button(root, text="Browse", command=browse_file)
browse_button.pack()

# Predict Button
predict_button = Button(root, text="Predict", command=predict)
predict_button.pack()

# Result Text
result_text = Text(root, height=5, width=50)
result_text.insert(END, "Results will be displayed here.")
result_text.config(state='disabled')
result_text.pack()

# Scrollbar for Result Text
scrollbar = Scrollbar(root, command=result_text.yview)
scrollbar.pack(side='right', fill='y')
result_text['yscrollcommand'] = scrollbar.set

root.mainloop()