
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
import librosa
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Suppress warnings
warnings.filterwarnings("ignore")

# Load models and scaler
model = load_model("models/ArtificialNeuralNetwork_model.h5")
with open(file="models/Scaler.pkl", mode="rb") as file:
    scaler = pickle.load(file=file)

# Class labels
class_labels = ['Acrocephalus', 'Bubo', 'Caprimulgus', 'Emberiza', 'Ficedula', 'Glaucidium', 'Hippolais']

# Function to process audio
def audio_to_signal(path):
    y, sr = librosa.load(path, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
        
    return to_append.split()

# GUI Implementation
def select_file():
    global audio_path
    audio_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3"), ("All Files", "*.*"))
    )
    if audio_path:
        file_label.config(text=f"Selected: {os.path.basename(audio_path)}")

def predict_audio():
    if not audio_path:
        messagebox.showerror("Error", "Please select an audio file first!")
        return
    try:
        # Preprocess the audio
        input_data = audio_to_signal(audio_path)
        input_data = [float(x) for x in input_data]
        input_data = np.array([input_data])
        scaled_data = scaler.transform(input_data)
        
        # Predict using the model
        prediction = model.predict(scaled_data, verbose=0)
        class_label = np.argmax(prediction)
        class_name = class_labels[class_label]
        probability = prediction[0][class_label]
        
        # Display results
        result_label.config(text=f"Prediction: {class_name} (Confidence: {probability:.2f})")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main window
root = tk.Tk()
root.title("Audio Classifier")
root.geometry("500x300")
root.configure(bg="#f0f8ff")

# Heading
heading = tk.Label(root, text="Audio Classifier", font=("Helvetica", 18, "bold"), bg="#4682b4", fg="white")
heading.pack(fill=tk.X)

# File Selection Section
file_frame = tk.Frame(root, bg="#f0f8ff")
file_frame.pack(pady=20)

file_button = tk.Button(file_frame, text="Select Audio File", command=select_file, bg="#4682b4", fg="white", font=("Arial", 12))
file_button.pack(side=tk.LEFT, padx=10)

file_label = tk.Label(file_frame, text="No file selected", bg="#f0f8ff", font=("Arial", 12))
file_label.pack(side=tk.LEFT)

# Prediction Section
predict_button = tk.Button(root, text="Predict", command=predict_audio, bg="#32cd32", fg="white", font=("Arial", 14))
predict_button.pack(pady=20)

result_label = tk.Label(root, text="Prediction: None", bg="#f0f8ff", font=("Arial", 12))
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
