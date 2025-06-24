Emotion Classification of Speech Data
A deep learning project for classifying emotions in speech and song audio using the RAVDESS dataset. The app is deployed on Streamlit:
Try the App

Overview
This project implements a 1D Convolutional Neural Network (CNN) to recognize emotions from audio files. Both speech and song samples are processed to classify into one of eight emotions:
neutral, calm, happy, sad, angry, fear, disgust, surprise. 

Features
Audio Input: Supports both speech and song samples from the RAVDESS dataset.

Emotion Classes: Classifies into 8 distinct emotions.

Deep Learning Model: Utilizes a 1D CNN with regularization and batch normalization for robust performance.

Interactive Web App: User-friendly Streamlit interface for uploading audio and viewing predictions.

Dataset
Source: [RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song]

Structure:

Audio_Speech_Actors_01-24/

Audio_Song_Actors_01-24/

Samples: 1,440 speech samples, organized by actor.

Model Architecture
Layers: Multiple Conv1D, BatchNormalization, MaxPooling, and Dropout layers.

Activation: ReLU in hidden layers, Softmax for output.

Optimizer: Adam (learning rate 0.001)

Loss: Categorical Crossentropy

Regularization: Dropout and L2 kernel regularization

Training: Early stopping and learning rate reduction on plateau, batch size 64, validation split 15%.

Results
Best Validation Accuracy: ~68%

Strengths: Good generalization across most emotion classes.

Limitations: Some confusion between acoustically similar emotions (e.g., fear vs. sad).

Usage
Clone the Repository
Place the RAVDESS dataset in the specified directories.

Install Requirements

text
pip install -r requirements.txt
Run the App

text
streamlit run app.py
Or visit the Streamlit deployment.

Key Files
app.py — Streamlit web app

emotion_classifier_model.h5 / .keras — Trained model weights

mars_emotion_class.ipynb — Jupyter notebook for data processing and model training

features.csv, speech_data.csv — Processed features and labels

References
RAVDESS Dataset: Livingstone, S.R., & Russo, F.A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

For more details, see the code and documentation in this repository.
Try the app live: https://emotion-classification-of-speech-data.streamlit.app/
