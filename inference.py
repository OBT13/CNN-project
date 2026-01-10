import torch
from torchvision import transforms, models
from PIL import Image
import os

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Inference on:", device)

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =========================
# LOAD MODEL
# =========================
num_classes = 28
model = models.resnet18(pretrained=False)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_arabic_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# =========================
# CLASS LABELS
# =========================
# labels من 1 → 28
class_labels = list(range(1, 29))

# =========================
# FUNCTION: predict single image
# =========================
def predict_image(img_path):
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return class_labels[pred.item()]

# =========================
# EXAMPLES
# =========================
# 1️⃣ Predict single image
image_path = "youness.png"  # بدل بالمسار ديال الصورة
predicted_class = predict_image(image_path)
print(f"Prediction for {image_path}: {predicted_class}")

# 2️⃣ Predict all images in a folder
# folder_path = "data/test_images_folder"  # بدل بالمسار ديال المجلد
# for filename in os.listdir(folder_path):
#     if filename.endswith((".png", ".jpg", ".jpeg")):
#         img_path = os.path.join(folder_path, filename)
#         pred_class = predict_image(img_path)
#         print(f"{filename} → Predicted Class: {pred_class}")
