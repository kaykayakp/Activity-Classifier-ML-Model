# Activity Classifier – Walking vs. Jumping 

A machine learning project to classify human motion (walking vs. jumping) using accelerometer data collected from smartphones, built with Python, scikit-learn, and deployed as a desktop GUI application.

## Overview
This project uses raw sensor data (X, Y, Z axes) collected via the Phyphox app to train a logistic regression classifier. The model achieves 100% test accuracy and is deployed in a Tkinter-based GUI for user interaction. A real-time classification pipeline was also developed as a bonus feature.

## Tech Stack
- **Languages**: Python
- **Libraries**: scikit-learn, pandas, NumPy, SciPy, Matplotlib, Joblib, H5Py, Tkinter, Selenium
- **Tools**: Phyphox (mobile sensor app), Git, Jupyter, VS Code

##  File Structure
- `dataset.h5`: Consolidated data including raw, filtered, and segmented acceleration values
- `model_training.py`: Data preprocessing, segmentation, feature extraction, model training + evaluation
- `gui_app.py`: Standalone desktop app with file upload, classifier output, and plotted results
- `bonus.py`: Real-time classifier using Selenium to stream live accelerometer data from phone
- `activity_classifier.pkl`: Final saved logistic regression model
- `processed/`, `raw/`, `segmented/`: Folder structure for organizing CSV input and feature files

##  Project Pipeline

### 1. **Data Collection**
- 5–6 minutes of walking and jumping recorded per participant
- Data collected via **Phyphox** app in various phone placements and postures (pockets, hands, etc.)
- Exported as `.csv` and manually labeled by activity and participant

### 2. **Preprocessing**
- Applied moving average filter (window size = 51) to reduce high-frequency noise
- Signals smoothed while retaining key motion patterns (e.g., walking periodicity, jumping spikes)
- Stored clean data in both CSV and HDF5 formats

### 3. **Feature Extraction**
- Extracted 10 statistical features (mean, std, RMS, kurtosis, etc.) from each 5s segment
- Used only absolute acceleration features for better orientation invariance
- Standardized features using `StandardScaler` for model compatibility

### 4. **Model Training**
- Trained **Logistic Regression** classifier using scikit-learn
- 90/10 train-test split; pipeline included scaling + training
- Achieved 100% accuracy, recall, and AUC on test set  
- Model saved to `activity_classifier.pkl`

### 5. **Model Deployment – GUI App**
- Built a user-facing **Tkinter app** to upload CSV, classify activity, and visualize predictions
- Segments accelerometer data, extracts features, and returns walking/jumping predictions
- Outputs predictions in both tabular and graphical form with red/blue highlights

### 6. **Bonus: Real-Time Classification**
- Connected to **live Phyphox data stream** via Selenium scraping
- Performed in-memory feature extraction and live classification every 20 samples
- Terminal-based interface showed segment-wise predictions in real time

##  Model Evaluation
- **Confusion Matrix**: 100% precision and recall
- **ROC AUC**: 1.00
- Correlation analysis confirmed most predictive features (e.g., RMS, variance of abs accel)

##  My Role
- Implemented preprocessing pipeline and feature extraction logic
- Built and trained ML model (logistic regression) using scikit-learn
- Developed the GUI-based desktop application with segmentation + live prediction output
- Created and tested real-time data scraping and classification via Selenium
