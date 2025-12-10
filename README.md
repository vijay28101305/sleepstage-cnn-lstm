# Automated Sleep Stage Classification using CNN–LSTM on EEG (Sleep-EDF)

This project implements an end-to-end deep learning pipeline for classifying sleep stages from raw EEG signals using the Sleep-EDF Expanded dataset. The model combines CNNs for feature extraction and LSTMs for temporal modeling, achieving 82.09% accuracy on 5-stage sleep classification using only the Fpz–Cz EEG channel.

This work was developed as part of the University of Colorado Denver – MS Course Project (2025).

## Project Overview

Manual sleep staging requires trained technicians and several hours of EEG inspection per night. This project automates the process by:

- Loading and preprocessing raw EEG recordings  
- Constructing 30-second epochs and context windows  
- Extracting temporal features using a CNN–LSTM model  
- Evaluating classification performance  
- Performing explainability analysis using classical EEG features  

## Results

**Test Accuracy:** 82.09%  
**Classes:** W, N1, N2, N3, REM  
**Dataset:** Sleep-EDF (Fpz–Cz channel)

### Confusion Matrix Summary
- N3 (deep sleep): highest accuracy due to strong delta-wave activity  
- N2: stable performance driven by spindle and K-complex detection  
- REM & N1: expected confusion due to similar low-amplitude mixed-frequency EEG  

## Model Architecture

### 1. CNN Feature Extractor
Captures local EEG characteristics such as sleep spindles, K-complexes, slow waves, and alpha/beta rhythms.

### 2. LSTM Model
Learns transitions between stages using a 5-epoch context window.

### 3. Fully Connected Classifier
Outputs probabilities over the 5 sleep-stage labels.

## Pipeline

1. Download Sleep-EDF dataset  
2. Bandpass filter EEG (0.3–35 Hz)  
3. Segment into 30-second epochs  
4. Normalize each epoch  
5. Map annotations to numerical labels  
6. Construct 5-epoch temporal sequences  
7. Train CNN–LSTM with class-balanced loss  
8. Evaluate on held-out subjects  
9. Perform classical EEG feature explainability analysis  

## Explainability

To validate the model’s behavior, classical EEG features were extracted:

- Delta / Theta / Alpha / Beta / Sigma band power  
- Spectral entropy  
- Zero-crossings  
- RMS amplitude  
- Band ratios  

### Findings
- Delta power and zero-crossings showed highest importance (ANOVA F-score)  
- PCA and t-SNE demonstrated clear clustering between sleep stages  
- Correlation heatmap confirmed known physiological relationships (e.g., high delta in N3, reduced alpha/beta in deeper sleep)

## Repository Structure
```
sleep-stage-cnn-lstm/
├── Machine_learning_project.ipynb # Main training + analysis notebook
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── images/ # Optional figures for explainability
```

## How to Run

### Install dependencies:
pip install -r requirements.txt


### Run the notebook:
Open `Machine_learning_project.ipynb` and execute all cells.

## Future Improvements

- Use multi-channel EEG (Fpz–Cz + Pz–Oz + EOG/EMG)  
- Increase temporal context to 15–30 epochs  
- Incorporate transformer-based sequence models  
- Add EEG artifact detection and removal  
- Explore subject-adaptive fine-tuning  

## Elevator Pitch

This system automatically scores sleep stages from raw EEG, reducing clinical scoring time from hours to seconds and enabling scalable sleep diagnostics.
