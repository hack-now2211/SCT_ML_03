This project is a Cat and Dog Classifier web application that uses Machine Learning and Deep Learning to classify images of cats and dogs. It utilizes TensorFlow to leverage GPU processing for efficient image feature extraction, and Support Vector Machine (SVM) for classification.

Key Features
Frontend Interface: An interactive, user-friendly web page built with HTML and Flask that allows users to upload an image. Upon upload, it predicts if the image is of a cat or a dog, displaying the result along with prediction accuracy.
Model Backend: A pretrained MobileNetV2 model extracts features from the image, which are then classified by a Support Vector Machine (SVM) model. The model was trained using separate folders for cats and dogs in training, validation, and testing datasets.
Real-Time Prediction: Users can upload their own images to receive real-time predictions with an accuracy score, making it a functional demo of image classification.
This project provides a practical example of applying image processing, feature extraction, and machine learning techniques to create a predictive web application.
