Streamlit Link--    https://emotion-classification-of-speech-data.streamlit.app/

Overview
This project implements a deep learning pipeline for speech emotion recognition using audio data from the RAVDESS dataset.
The notebook processes both speech and song audio files, extracts relevant features, and trains a 1D Convolutional Neural Network (CNN) to classify emotions into eight categories. 
The solution leverages Python, popular data science libraries, and TensorFlow/Keras for model building and training1.

Dataset
Source: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Structure: Audio files are organized by actor in two directories:
          ./Audio_Speech_Actors_01-24/
          ./Audio_Song_Actors_01-24/

Emotions Covered:   neutral,calm,happy,sad,angry,fear,disgust,surprise

Total Speech Samples: 1,440

Data Preprocessing

Directory Parsing: The notebook lists actor directories and iterates through .wav files, extracting emotion labels from file names by parsing the filename convention1.
DataFrames: Paths and emotion labels are stored in pandas DataFrames for further processing.
Label Mapping: Numeric emotion codes are mapped to string labels (e.g., 1 → neutral, 2 → calm, etc.)1.
Feature Extraction: (Not fully shown in the excerpt) Typically, MFCCs or similar audio features are extracted for each sample.


Model Architecture
The model is a deep 1D CNN designed for sequential audio data:

Layer Type	Output Shape	Parameters
Conv1D (128 filters)	(None, 162, 128)	768
BatchNormalization	(None, 162, 128)	512
MaxPooling1D	(None, 41, 128)	0
Conv1D (256 filters)	(None, 41, 256)	164,096
BatchNormalization	(None, 41, 256)	1,024
MaxPooling1D	(None, 11, 256)	0
Conv1D (128 filters)	(None, 11, 128)	98,432
BatchNormalization	(None, 11, 128)	512
Dropout (0.3)	(None, 11, 128)	0
MaxPooling1D	(None, 3, 128)	0
Conv1D (64 filters)	(None, 3, 64)	24,640
BatchNormalization	(None, 3, 64)	256
Dropout (0.3)	(None, 3, 64)	0
GlobalAveragePooling1D	(None, 64)	0
Dense (64 units, relu)	(None, 64)	4,160
Dropout (0.4)	(None, 64)	0
Dense (8 units, softmax)	(None, 8)	520
Total Parameters: ~295,000

Activation Functions: ReLU (hidden layers), Softmax (output)
Regularization: Dropout and L2 kernel regularization
Optimizer: Adam (learning rate 0.001)
Loss Function: Categorical Crossentropy
Metrics: Accuracy1

Training
Epochs: Up to 60, with early stopping and learning rate reduction on plateau.
Batch Size: 64
Validation Split: 15%
Class Weights: Used to handle class imbalance.
Callbacks: EarlyStopping and ReduceLROnPlateau for robust training1.
Example Training Progress
Rapid improvement in accuracy and loss over the first 20 epochs.
Early stopping typically triggers around epoch 32, with the best validation accuracy observed near 68%1.

Evaluation
Metrics: Accuracy and loss are tracked for both training and validation sets.

Confusion Matrix: Predicted vs. actual emotion labels are compared to assess model performance.

Example Results: The model demonstrates strong performance on several emotion classes, with some confusion between similar emotions (e.g., fear vs. sad)1.

Usage
Requirements
Python ,pandas, numpy, matplotlib, seaborn,librosa,TensorFlow/Keras

Running the Notebook
Place the RAVDESS dataset in the expected directory structure.

Run all cells in the notebook to preprocess data, extract features, train the model, and evaluate results.

Key Files
mars_emotion_class-6.ipynb: Main Jupyter notebook containing all code for data processing, model training, and evaluation1.
/Audio_Speech_Actors_01-24/, ./Audio_Song_Actors_01-24/: Directories containing audio data.

Results: Best Validation Accuracy: ~68% (varies depending on run and hyperparameters)
Strengths: Good generalization across most emotion classes.
Limitations: Some confusion between acoustically similar emotions; further improvements possible with data augmentation or advanced architectures1.

References
RAVDESS Dataset: Livingstone, S.R., & Russo, F.A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

mars_emotion_class-6.ipynb
