"""
Author: Annam.ai IIT Ropar
Team Members: Aman Sagar
Leaderboard Rank: 6

"""

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

# Install required packages
!pip install -q torch torchvision pandas scikit-learn

# Imports
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Preprocessing
df = pd.read_csv(TRAIN_CSV)
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Dataset
class BinarySoilDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, test=False):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image, img_id
        else:
            label = self.df.iloc[idx]['label']
            return image, label


# Loaders
BATCH_SIZE = 64

train_dataset = BinarySoilDataset(df_train, TRAIN_DIR, transform=train_transforms)
val_dataset = BinarySoilDataset(df_val, TRAIN_DIR, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

test_df = pd.read_csv(TEST_CSV)
test_dataset = BinarySoilDataset(test_df, TEST_DIR, transform=val_transforms, test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)





