import torch
from model import SmallCNN
from PIL import Image
import torchvision.transforms as transforms

arabic_letters = [
    "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز",
    "س","ش","ص","ض","ط","ظ","ع","غ","ف","ق",
    "ك","ل","م","ن","ه","و","ي"
]

model = SmallCNN()
model.load_state_dict(torch.load("arabic_cnn.pth"))
model.eval()

img = Image.open("test.png").convert("L")
img = img.resize((32,32))

transform = transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(1).item()

print("Model says:", arabic_letters[pred])
