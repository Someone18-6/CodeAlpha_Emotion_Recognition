# Speech Emotion Recognition Using Machine Learning

**Speech Emotion Recognition** is a machine learning project that predicts emotions from speech audio using audio feature extraction and an XGBoost classifier. The project includes an interactive web application built with Streamlit.

Developed as part of my **CodeAlpha Internship**.

---

## Project Overview

Human speech contains information beyond words, including tone, pitch, and acoustic patterns. This project explores how machine learning can use these audio characteristics to classify speech into different emotional categories.

The model is trained using the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

## Features

- Upload audio recordings for emotion prediction.
- Predict one of eight emotion classes.
- Extract audio features using Librosa.
- Classify emotions using an XGBoost model.
- Interact with the model through a Streamlit web application.
- View the predicted emotion through the app interface.

## Emotion Classes

The model predicts the following eight emotions:

- **Angry**
- **Calm**
- **Disgust**
- **Fearful**
- **Happy**
- **Neutral**
- **Sad**
- **Surprised**

## Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Librosa** | Audio processing and feature extraction |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Machine learning utilities |
| **XGBoost** | Emotion classification |
| **Streamlit** | Interactive web application |
| **Joblib** | Saving and loading trained model artifacts |

## Audio Feature Extraction

The system extracts three types of audio features to capture useful characteristics of speech.

- **MFCC (Mel-Frequency Cepstral Coefficients):** Captures characteristics of the audio's spectral envelope.
- **Chroma Features:** Represents the distribution of audio energy across pitch classes.
- **Mel Spectrogram:** Represents how audio energy varies across frequencies over time.

The mean and standard deviation of these features are calculated and combined into a **360-dimensional feature vector** for prediction.

## Model and Evaluation

An **XGBoost classifier** is trained using features extracted from the RAVDESS dataset.

**Test Accuracy: 68.06% across eight emotion classes.**

This result reflects the model's performance on the test data used in the project. Real-world performance may vary depending on recording quality, background noise, and differences between speakers.

## Project Structure

```text
Emotion_Recognition/
│
├── Audio_Speech_Actors_01-24.zip
├── app_emo_recg.py
├── app_emo_recg_backup.py
├── app.py
│
├── processed_data/
│   ├── features.npy
│   ├── labels.npy
│   ├── emotion_features.csv
│   ├── features_improved.npy
│   └── labels_improved.npy
│
├── ravdess/
│   ├── Actor_01/
│   ├── Actor_02/
│   ├── Actor_03/
│   ├── ...
│   └── Actor_24/
│
├── model/
│   ├── emotion_xgboost.joblib
│   ├── label_encoder.joblib
│   └── metadata.json
│
├── .ipynb_checkpoints/
│
└── .streamlit/
    └── config.toml
```

**Folder and File Description**

- **`app_emo_recg.py`** — Main Streamlit application for speech emotion prediction.
- **`app_emo_recg_backup.py`** — Backup copy of the application.
- **`app.py`** — Additional application script.
- **`Audio_Speech_Actors_01-24.zip`** — Compressed RAVDESS audio dataset.
- **`ravdess/`** — Extracted dataset containing audio recordings organized into 24 actor folders.
- **`processed_data/`** — Stores extracted audio features and corresponding labels in NumPy and CSV formats.
- **`model/`** — Contains the trained XGBoost classifier, label encoder, and model metadata.
- **`.streamlit/config.toml`** — Contains the Streamlit application's theme configuration.
- **`.ipynb_checkpoints/`** — Automatically generated checkpoint files from Jupyter or Colab notebooks.
## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Someone18-6/CodeAlpha_Emotion_Recognition.git
cd CodeAlpha_Emotion_Recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that the `requirements.txt` file includes all packages required by the application, such as Streamlit, Librosa, NumPy, scikit-learn, XGBoost, and Joblib.

### 4. Run the Application

```bash
streamlit run app_emo_recg.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### 5. Make a Prediction

1. Open the Streamlit application.
2. Upload a supported audio recording.
3. Run the prediction through the app interface.
4. View the predicted emotion.

**Note:** The trained model artifacts must be available at the paths expected by the application.

## Results

The model achieved **68.06% test accuracy** on the eight-class emotion classification task.

This project provided hands-on experience with the complete machine learning workflow, including audio feature extraction, model training, evaluation, and application development.

## Future Improvements

- Explore deep learning approaches such as **CNNs and LSTMs**.
- Experiment with improved audio preprocessing and feature extraction.
- Evaluate performance on additional speakers and unseen audio recordings.
- Improve robustness to background noise and variations in recording quality.
- Compare different classification models and evaluation metrics.

## Dataset

**RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song**

The dataset contains acted emotional speech and song recordings.

**Dataset Link:** https://zenodo.org/records/1188976

Please follow the dataset's applicable terms and conditions when using or redistributing its contents.

## Author

**Karanam Shasank**

- **GitHub:** https://github.com/Someone18-6
- **LinkedIn:** https://www.linkedin.com/in/karanam-shasank-3a4355329/

## Acknowledgements

Thanks to **CodeAlpha** for the internship opportunity and learning experience.

Thanks to the creators of the RAVDESS dataset and the developers of the open-source libraries used in this project.
