#!/bin/bash

# Google Drive folder ID
FOLDER_ID="1juuj0X3O5aEqxF4xaHHV7DhNWy0bo3pl"
TARGET_DIR="./data"

echo "📁 Downloading folder from Google Drive (ID: $FOLDER_ID)"
mkdir -p "$TARGET_DIR"

# Check for gdown
if ! command -v gdown &> /dev/null
then
    echo "❌ 'gdown' not found. Install it via: pip install gdown"
    exit 1
fi

# Download the folder
gdown --folder "$FOLDER_ID" -O "$TARGET_DIR"

echo "✅ Download complete. Files saved to: $TARGET_DIR"
