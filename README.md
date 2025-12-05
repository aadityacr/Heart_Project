# ECG-Based Heart Disease Prediction System

This project implements a Deep Learning framework to classify cardiac abnormalities using Electrocardiogram (ECG) waveform data. The system analyzes PQRST wave patterns to detect specific heart conditions with high precision, aiming to assist in automated medical diagnostics.

## Project Overview
The primary objective of this repository is to provide a robust model for classifying ECG signals. By leveraging neural networks, the system identifies distinct patterns associated with various cardiac pathologies, offering a non-invasive tool for early diagnosis.

## Classification Categories
The model (`Five_Class_Model.h5`) has been trained to distinguish between the following five cardiac conditions:

1. **Normal Sinus Rhythm** (Baseline/Healthy)
2. **Arrhythmia** (Irregular heart rhythm)
3. **Atrial Fibrillation** (Rapid, irregular atrial rhythm)
4. **Myocardial Infarction** (Heart attack)
5. **ST Depression** (Indicator of myocardial ischemia)

## Repository Structure
*   **`app.py`**: The primary application script serving as the interface for the prediction model.
*   **`Five_Class_Model.h5`**: The pre-trained Deep Learning model weights.
*   **`FIGURES/`**: Directory containing performance visualization assets.
*   **`archive/`**: Source dataset and raw signal files (excluded from version control due to size constraints).

## Performance Evaluation
The repository includes comprehensive performance metrics located in the root directory. These include confusion matrices and accuracy graphs that demonstrate the model's efficacy across all five target classes.

## Technical Dependencies
*   **Language:** Python 3.x
*   **Framework:** TensorFlow / Keras
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib / Seaborn