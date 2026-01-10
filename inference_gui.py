import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tkinter import Tk, filedialog

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
# ARABIC LETTERS
# =========================
arabic_letters = [
    "أ","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص",
    "ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي"
]

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(img_path):
    img = Image.open(img_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, k=3)
    
    top3_prob = top3_prob.cpu().squeeze().numpy()
    top3_idx = top3_idx.cpu().squeeze().numpy()
    top3_labels = [arabic_letters[i] for i in top3_idx]
    
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Pred: {top3_labels[0]}\nTop3: {top3_labels} ({top3_prob.round(2)})")
    plt.show()

# =========================
# FILE DIALOG
# =========================
root = Tk()
root.withdraw()  # نخفي نافذة tkinter الرئيسية

file_path = filedialog.askopenfilename(
    title="اختر صورة للحروف العربية",
    filetypes=[("Images", "*.png *.jpg *.jpeg")]
)

if file_path:
    predict_image(file_path)
else:
    print("لم يتم اختيار أي صورة.")
