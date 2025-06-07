from flask import Flask, render_template, request, flash, redirect
import warnings
import librosa
import sqlite3
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import shutil
import os
from werkzeug.utils import secure_filename

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)


# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'audio_classifier_secret_key'

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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('main.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route("/main", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        file = request.files['audio_file']
        filename = secure_filename(file.filename)
        file_path = 'static/temp/'+filename
        file.save(file_path)
        try:
            # Preprocess the audio
            input_data = audio_to_signal(file_path)
            input_data = [float(x) for x in input_data]
            input_data = np.array([input_data])
            scaled_data = scaler.transform(input_data)
            
            # Predict using the model
            prediction = model.predict(scaled_data, verbose=0)
            class_label = np.argmax(prediction)
            class_name = class_labels[class_label]
            probability = prediction[0][class_label]
            print(prediction)
            print(class_name)
            
            return render_template("main.html",audio_url='static/temp/'+filename, img = f'http://127.0.0.1:5000/static/img/{class_name}.jpeg', result=f"{class_name} (Confidence: {probability:.2f})")
        except Exception as e:
            print(f"Error processing the file: {str(e)}")
            return redirect(request.url)
    
    return render_template("main.html", result=None)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
