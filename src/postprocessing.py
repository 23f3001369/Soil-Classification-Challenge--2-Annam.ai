"""
Author: Annam.ai IIT Ropar
Team Members: Aman Sagar
Leaderboard Rank: 16

"""

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

class EarlyStopping:
    def __init__(self, patience=2, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping(patience=2)
EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break


model.eval()
all_preds, all_ids = [], []

with torch.no_grad():
    for inputs, img_ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).squeeze(1)
        preds = (probs > 0.5).long().cpu().numpy()  # Adjust threshold if necessary
        all_preds.extend(preds)
        all_ids.extend(img_ids)

import cv2

def calculate_blur(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(image, cv2.CV_64F).var()


BLUR_THRESHOLD = 100

# Apply blur logic
final_preds = []
for img_id, resnet_pred in zip(all_ids, all_preds):
    img_path = os.path.join(TEST_DIR, img_id)
    blur_score = calculate_blur(img_path)

    if blur_score < BLUR_THRESHOLD:
        final_preds.append(0)  # Very blurry → likely non-soil
    else:
        final_preds.append(resnet_pred)  # Trust model if not blurry


submission = pd.DataFrame({
    "image_id": all_ids,
    "label": final_preds
})
submission.to_csv("submission-part2_blur_corrected.csv", index=False)
print(" Saved: submission-part2_blur_corrected.csv")

from google.colab import files
files.download('submission-part2_blur_corrected.csv')
