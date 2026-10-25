Speech Emotion Recognition Using Machine Learning

An AI/ML project that predicts emotions from speech audio using audio feature extraction and an XGBoost classifier. The project includes an interactive web application built with Streamlit, allowing users to upload audio and view the predicted emotion.

Developed as part of my CodeAlpha internship.

Project Overview

Human speech contains information beyond words, including tone, pitch, and other acoustic patterns. This project explores how machine learning can use audio features to classify speech into different emotional categories.

The model is trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Features

- Upload an audio recording for emotion prediction.
- Predict one of eight emotion classes.
- Extract audio features using Librosa.
- Use an XGBoost classifier for prediction.
- Interact with the model through a Streamlit web application.
- View the predicted emotion in the app.

Emotion Classes

The model predicts the following eight emotions:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

Technologies Used

Technology| Purpose
Python| Core programming language
Librosa| Audio processing and feature extraction
NumPy| Numerical operations
Scikit-learn| Label encoding and machine learning utilities
XGBoost| Emotion classification model
Streamlit| Interactive web application
Joblib| Saving and loading trained model artifacts

Audio Feature Extraction

The system extracts audio features that help represent different characteristics of speech.

- MFCC (Mel-Frequency Cepstral Coefficients): Represents important characteristics of the audio's spectral envelope.
- Chroma Features: Represent the distribution of audio energy across pitch classes.
- Mel Spectrogram: Represents how audio energy varies across frequencies over time.

For each feature group, statistical summaries such as the mean and standard deviation are calculated and combined into a 360-dimensional feature vector for model prediction.

Model and Evaluation

An XGBoost classifier is trained using features extracted from the RAVDESS dataset.

Reported test accuracy: 68.06% across eight emotion classes.

This result reflects the model's evaluation on the test data used in this project. Performance on real-world recordings may differ depending on recording quality, background noise, speakers, and other factors.

Project Structure

CodeAlpha_Emotion_Recognition/
│
├── app_emo_recg.py
├── model/
│   ├── emotion_xgboost.joblib
│   ├── label_encoder.joblib
│   └── metadata.json
│
├── processed_data/
├── .streamlit/
│   └── config.toml
├── README.md
└── requirements.txt

Note: This structure represents the main files and folders used by the application. Your repository may contain additional files. The dataset and model artifacts should be included or obtained according to the setup instructions.

Getting Started

1. Clone the Repository

git clone https://github.com/Someone18-6/CodeAlpha_Emotion_Recognition.git
cd CodeAlpha_Emotion_Recognition

2. Create a Virtual Environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

On macOS or Linux:

source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

Ensure that "requirements.txt" includes the packages required by the application, including Streamlit, Librosa, NumPy, scikit-learn, XGBoost, and Joblib.

4. Run the Application

streamlit run app_emo_recg.py

Open the local URL shown in your terminal, usually:

http://localhost:8501

5. Make a Prediction

1. Open the Streamlit application.
2. Upload a supported audio recording.
3. Run the prediction using the app's interface.
4. View the predicted emotion.

The trained model artifacts must be available at the paths expected by the application. Check the app's requirements for accepted audio formats and sampling conditions.

Results

The trained model achieved 68.06% test accuracy on the project's eight-class emotion classification task.

This project demonstrates a complete machine learning workflow, from extracting features from raw audio to evaluating a classifier and integrating it into a web application.

Future Improvements

- Explore CNN- and LSTM-based deep learning models.
- Experiment with improved audio preprocessing and feature extraction.
- Evaluate performance using additional speakers and unseen audio recordings.
- Improve robustness to background noise and variations in recording quality.
- Compare different classification models and evaluation metrics.

Dataset

RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song

The dataset contains acted emotional speech and song recordings.

Dataset website: https://zenodo.org/records/1188976

Please follow the dataset's applicable terms and conditions when downloading, using, or redistributing its contents.

Author

Karanam Shasank

- GitHub: https://github.com/Someone18-6
- LinkedIn: https://www.linkedin.com/in/karanam-shasank-3a4355329/

Acknowledgements

Thanks to CodeAlpha for the internship opportunity and learning experience.

Thanks to the creators of the RAVDESS dataset and the developers of the open-source libraries used in this project.
