# Automated Sleep Stage Classification using CNN–LSTM on EEG (Sleep-EDF)

This repository contains an end-to-end deep learning pipeline for sleep stage classification from EEG using the Sleep-EDF Expanded dataset, and also includes a deployed CLI inference pipeline with frozen model weights for reproducible demo and evaluation.

The model combines CNNs for feature extraction and LSTMs for temporal modeling, achieving 82.09% accuracy on 5-stage sleep classification using only the Fpz–Cz EEG channel.

This work was developed as part of the University of Colorado Denver – MS Course Project (2025).

## Deployed Inference (CLI)

This repository is ready for inference using a command-line script.  
No training is required to run the demo.

### What is included in deployment
- Frozen model weights (sleepnet_deploy/model_weights.pth)
- Batch inference via CLI (infer_cli.py)
- Confusion matrix computation
- CSV export of predictions
- Demo input file (sample.npz)

### Run inference (demo)

Install dependencies:

    pip install -r requirements.txt

Run CLI inference on the demo EEG file:

    python infer_cli.py \
      --input sample.npz \
      --weights sleepnet_deploy/model_weights.pth \
      --config sleepnet_deploy/config.json \
      --label-map sleepnet_deploy/label_map.json

## Project Overview (Research Context)

Manual sleep staging requires trained technicians and several hours of EEG inspection per night. This project automates the process by:

- Loading and preprocessing raw EEG recordings
- Constructing 30-second epochs and context windows
- Extracting temporal features using a CNN–LSTM model
- Evaluating classification performance
- Performing explainability analysis using classical EEG features

## Results

Test Accuracy: 82.09%  
Classes: W, N1, N2, N3, REM  
Dataset: Sleep-EDF (Fpz–Cz channel)

### Confusion Matrix Summary
- N3 (deep sleep): highest accuracy due to strong delta-wave activity
- N2: stable performance driven by spindle and K-complex detection
- REM & N1: expected confusion due to similar low-amplitude mixed-frequency EEG

## Model Architecture

### CNN Feature Extractor
Captures local EEG characteristics such as sleep spindles, K-complexes, slow waves, and alpha/beta rhythms.

### LSTM Model
Learns transitions between stages using a 5-epoch context window.

### Fully Connected Classifier
Outputs probabilities over the 5 sleep-stage labels.

## Pipeline (Training)

1. Download Sleep-EDF dataset
2. Bandpass filter EEG (0.3–35 Hz)
3. Segment into 30-second epochs
4. Normalize each epoch
5. Map annotations to numerical labels
6. Construct 5-epoch temporal sequences
7. Train CNN–LSTM with class-balanced loss
8. Evaluate on held-out subjects
9. Perform classical EEG feature explainability analysis

## Repository Structure

    sleepstage-cnn-lstm/
    ├── infer_cli.py
    ├── model.py
    ├── sample.npz
    ├── sleepnet_deploy/
    │   ├── model_weights.pth
    │   ├── config.json
    │   └── label_map.json
    ├── notebooks/
    │   └── Machine_learning_project.ipynb
    ├── requirements.txt
    └── README.md

## Elevator Pitch

This system automatically scores sleep stages from raw EEG, reducing clinical scoring time from hours to seconds and enabling scalable sleep diagnostics.
