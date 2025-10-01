# EEG_Artifact_Removal
This repository contains the implementation of my master’s thesis project: a deep learning approach for removing artifacts in low-channel EEG signals. The work explores and compares different architectures designed to improve EEG signal quality, which is crucial for clinical and research applications, especially in portable and long-term monitoring systems.

# Repository Structure

Utils/ – Utility functions required for the implementation of the main scripts.

train/-Srcipt that conatains the trainer.

models/DAE_models.py – Contains all models used in the project, including exploratory architectures and different versions of the 1D SE-UNet with varying depths.

models/cnn_models.py – Includes all CNN-based models used for grid search experiments and the study of the 1D SE-ResNet architecture.

# Getting Started

For a complete understanding of the workflow, start with the main grid search scripts.
These scripts were used to identify the best-performing configurations of 1D SE-UNet and 1D SE-ResNet, enabling a systematic evaluation of their components and depth variations.
