import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os
import cv2
train_dir = 'C:/Users/Venkatesh/OneDrive/Desktop/Internships/Skillcraft internship/Task 3/dataset/train'
test_dir = 'C:/Users/Venkatesh/OneDrive/Desktop/Internships/Skillcraft internship/Task 3/dataset/test'
validation_dir = 'C:/Users/Venkatesh/OneDrive/Desktop/Internships/Skillcraft internship/Task 3/dataset/validation'
img_size = 224  # Size for MobileNetV2 input
batch_size = 32
def extract_features(directory, model, img_size=224):
    data_gen = ImageDataGenerator(rescale=1./255)
    data = data_gen.flow_from_directory(
        directory,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    features = []
    labels = []
    
    for inputs_batch, labels_batch in data:
        features_batch = model.predict(inputs_batch)
        features.append(features_batch)
        labels.append(labels_batch)
        
        # Break loop after getting all data
        if len(features) * batch_size >= data.samples:
            break
    
    return np.vstack(features), np.hstack(labels)

# Load MobileNetV2 for feature extraction, removing the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))

# Extract features for training, validation, and test sets
train_features, train_labels = extract_features(train_dir, base_model)
val_features, val_labels = extract_features(validation_dir, base_model)
test_features, test_labels = extract_features(test_dir, base_model)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(train_features, train_labels)

# Save the model
joblib.dump(svm_model, 'cat_dog_svm_model.pkl')
# Validation Accuracy
val_predictions = svm_model.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Test Accuracy
test_predictions = svm_model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
