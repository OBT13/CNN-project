import torch
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
# Labels من 1 → 28
class_labels = list(range(1, 29))

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(img_path, show_image=True):
    img = Image.open(img_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, k=3)
    
    top3_prob = top3_prob.cpu().squeeze().numpy()
    top3_idx = top3_idx.cpu().squeeze().numpy()
    top3_labels = [class_labels[i] for i in top3_idx]
    
    if show_image:
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Pred: {top3_labels[0]}\nTop3: {top3_labels} ({top3_prob.round(2)})")
        plt.show()
    
    return top3_labels, top3_prob

# =========================
# SINGLE IMAGE EXAMPLE
# =========================
image_path = "youness.png"  # بدل بالمسار ديال الصورة
top3_labels, top3_prob = predict_image(image_path)
print(f"Prediction for {image_path}: {top3_labels} with probs {top3_prob}")

# =========================
# FOLDER EXAMPLE
# =========================
# folder_path = "data/test_images_folder"  # بدل بالمجلد ديالك
# if os.path.exists(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith((".png", ".jpg", ".jpeg")):
#             img_path = os.path.join(folder_path, filename)
#             top3_labels, top3_prob = predict_image(img_path)
#             print(f"{filename} → Top3 Predictions: {top3_labels} with probs {top3_prob}")
# else:
#     print(f"Folder {folder_path} does not exist. Only single image inference done.")
