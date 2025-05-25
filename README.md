# Soil-Classification-Challenge--2-Annam.ai
### Soil vs No Soil Classification Model

This project implements a deep learning model to classify images into soil types or detect the absence of soil. It is designed to help in soil analysis and classification tasks by identifying different soil categories or no soil in the input images.

## Project Overview

- **Goal:** Classify images into one of the soil types or detect no soil.
- **Soil Types:** Alluvial, Black, Clay, Red
- **No Soil:** Images with no visible soil
- **Model Backbone:** ResNet50 (with experiments on EfficientNet-B3a)
- **Framework:** PyTorch
- **Dataset:** Custom soil image dataset with annotated labels for soil types and no soil category.

## Features

- Image preprocessing and augmentation
- Training and validation pipeline
- Evaluation metrics: accuracy, F1-score
- Model checkpointing and inference script
- Optional test time augmentation (TTA) for improved predictions

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/23f3001369/Soil-Classification-Challenge--2-Annam.ai
    cd Soil-Classification-Challenge--2-Annam.ai
    ```

2. Create and activate a Python environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

Train the model on your dataset by running preprocessing.py and postprocessing.py (present in src folder).
