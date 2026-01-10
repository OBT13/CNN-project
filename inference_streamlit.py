import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Inference on: {device}")

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
# STREAMLIT INTERFACE
# =========================
st.title("Arabic Character Recognition (ResNet18)")

uploaded_file = st.file_uploader("اختر صورة لحرف عربي", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # قراءة الصورة
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="الصورة المختارة", use_column_width=True)

    # التحويل + prediction
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, k=3)
    
    top3_prob = top3_prob.cpu().squeeze().numpy()
    top3_idx = top3_idx.cpu().squeeze().numpy()
    top3_labels = [arabic_letters[i] for i in top3_idx]

    # عرض النتائج
    st.subheader(f"Prediction: {top3_labels[0]}")
    st.write("Top-3 Predictions:")
    for letter, prob in zip(top3_labels, top3_prob):
        st.write(f"{letter}: {prob*100:.2f}%")
