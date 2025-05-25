# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths
import os
ROOT_DIR = "/content/drive/MyDrive/soil-classification-part-2/soil_competition-2025"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
TRAIN_CSV = os.path.join(ROOT_DIR, "train_labels.csv")
TEST_CSV = os.path.join(ROOT_DIR, "test_ids.csv")
