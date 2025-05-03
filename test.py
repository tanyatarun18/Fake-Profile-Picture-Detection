import os
import cv2
import torch
from PIL import Image
from torchvision import transforms, models

MODEL_PATH = "forensic_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# DETECT ARTIFACTS
def detect_artifacts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

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

# PREDICT FUNCTION
def predict_image(image_path):
    model = ForensicClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    image_cv = cv2.imread(image_path)
    artifact_score = detect_artifacts(image_cv)
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    artifact_tensor = torch.tensor([[artifact_score]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor, artifact_tensor)
        _, pred = torch.max(output, 1)

    classes = ['real', 'fake']
    print(f"\nPredicted Class: {classes[pred.item()]}")
    return classes[pred.item()]

image_path = r"E:\CV\Dataset\Test\Fake\fake_28.jpg"  
predict_image(image_path)
