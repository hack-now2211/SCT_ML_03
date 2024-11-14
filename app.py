import os
from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained SVM model and MobileNetV2 for feature extraction
svm_model = joblib.load('cat_dog_svm_model.pkl')
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Predict function to process user-uploaded images
def predict_image(image_path, model, base_model, img_size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Extract features
    features = base_model.predict(img)

    # Predict using SVM model
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0]

    # Determine class and accuracy
    class_label = 'Cat' if prediction[0] == 0 else 'Dog'
    accuracy = max(probability) * 100  # Highest probability as percentage

    return class_label, accuracy

# Flask route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Make prediction
            label, accuracy = predict_image(file_path, svm_model, base_model)
            
            # Remove the file after prediction to save space
            os.remove(file_path)

            return render_template('index.html', label=label, accuracy=accuracy)

    return render_template('index.html', label=None)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
