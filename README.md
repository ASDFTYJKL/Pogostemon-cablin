# Transferable and Rapid Identification of Pogostemon Cablin Varieties Based on Hyperspectral Imaging and Deep Learning

This repository contains the code, data, and results related to the project on the identification of **Pogostemon cablin** varieties using hyperspectral imaging combined with deep learning techniques. The project aims to develop a fast, scalable, and transferable model for the classification and quality assessment of **Pogostemon cablin** varieties, leveraging the spectral characteristics captured by hyperspectral imaging.

## Project Overview

**Pogostemon cablin**, commonly known as patchouli, is a plant widely used in traditional medicine and the fragrance industry. Accurate identification of its varieties is crucial for quality control and maximizing its therapeutic and commercial value. This project uses advanced hyperspectral imaging and deep learning methodologies to achieve rapid and accurate classification of **Pogostemon cablin** varieties.

### Key Features

- **Rapid Identification**: Utilizing hyperspectral imaging combined with deep learning allows for quick and accurate identification of different **Pogostemon cablin** varieties.
  
- **Scalability**: The model can be easily extended to include new varieties as they are introduced, ensuring it remains relevant and effective.

- **Transferability**: The developed model can be applied to data collected under different instrumental conditions by optimizing the transfer learning strategy, making it highly adaptable across various scenarios.

## Directory Structure

```plaintext
├── data/                   # Directory containing hyperspectral data
│   ├── raw/                # Raw hyperspectral images
│   ├── processed/          # Preprocessed data ready for model input
│   └── labels/             # Ground truth labels for training and validation
├── models/                 # Trained deep learning models
│   └── pogostemon_model.pth  # Final trained model file
├── scripts/                # Python scripts for data processing, training, and evaluation
│   ├── preprocess.py       # Script for data preprocessing
│   ├── train.py            # Script for training the deep learning model
│   └── evaluate.py         # Script for model evaluation
├── results/                # Directory for storing results and analysis
│   ├── figures/            # Generated figures and plots
│   └── metrics/            # Performance metrics
└── README.md               # Project description and instructions
