from flask import Flask, render_template, request
import os
from evaluate import predict_video
from model import SimpleCNN
import torch
import shutil
from extract_faces import extract_faces_from_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'uploads/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

model = SimpleCNN()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['media_file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    extract_faces_from_video(path, PROCESSED_FOLDER)
    prediction = predict_video(model, PROCESSED_FOLDER)
    prediction, confidence = predict_video(model, PROCESSED_FOLDER)
    if prediction == -1:
        label = "Unable to detect"
    else:
        label = f"{'REAL' if prediction == 0 else 'FAKE'} ({confidence:.2f}% confident)"


    shutil.rmtree(PROCESSED_FOLDER)
    os.makedirs(PROCESSED_FOLDER)

    return render_template('index.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
