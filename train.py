import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = r"E:\CV\Dataset"
MODEL_PATH = "forensic_model.pth"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUSTOM DATASET
class FakeFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, category in enumerate(['real', 'fake']):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                self.image_paths.append(os.path.join(category_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def detect_artifacts(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap.var()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        artifact_score = self.detect_artifacts(image)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)
        artifact_tensor = torch.tensor([artifact_score], dtype=torch.float32)
        return image, artifact_tensor, label

# TRANSFORMS & DATALOADERS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_loader = DataLoader(FakeFaceDataset(os.path.join(DATA_DIR, 'train'), transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FakeFaceDataset(os.path.join(DATA_DIR, 'validation'), transform), batch_size=BATCH_SIZE, shuffle=False)

# MODEL
class ForensicClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(512 + 1, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x, forensic_feat):
        x = self.cnn(x)
        x = torch.cat((x, forensic_feat), dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# TRAINING & EVALUATION
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, forensic, labels in loader:
        images, forensic, labels = images.to(DEVICE), forensic.to(DEVICE), labels.to(DEVICE)
        outputs = model(images, forensic)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, forensic, labels in loader:
            images, forensic = images.to(DEVICE), forensic.to(DEVICE)
            outputs = model(images, forensic)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))


model = ForensicClassifier().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_model(model, train_loader, optimizer, criterion)
    print(f"Train Loss: {train_loss:.4f}")
    evaluate_model(model, val_loader)

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
