import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import StrongCNN
from dataset import ArabicDataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# 1. Data Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(20),        # دوران الصور
    transforms.RandomHorizontalFlip(),     # قلب الصور أفقياً
    transforms.RandomVerticalFlip(),       # قلب الصور رأسياً
    transforms.RandomResizedCrop(32),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # تحسين الألوان
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),     # قص صور عشوائي
    transforms.ToTensor(),                # تحويل الصور إلى Tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # تطبيع الصور
])

# 2. Device setup (CUDA/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

# 3. تحميل الداتا مع augmentations
train_dataset = ArabicDataset(
    "data/csvTrainImages 13440x1024.csv",
    "data/csvTrainLabel 13440x1.csv",
    transform=transform
)

# 4. DataLoader (تحميل البيانات على دفعات)
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0  # Windows-compatible with num_workers=0
)

# 5. إعداد الموديل
model = StrongCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Learning Rate Scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)  # تنخفض الـ learning rate بعد كل 10 epochs

# 7. Training loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()  # Update learning rate at each epoch
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

# 8. Save model
torch.save(model.state_dict(), "arabic_cnn_gpu.pth")
print("Model saved ✔️")

# 9. Test the model (evaluation on test data)
test_dataset = ArabicDataset(
    "data/csvTestImages 3360x1024.csv",
    "data/csvTestLabel 3360x1.csv",
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
model.eval()  # Set model to evaluation mode

correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation for testing
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy:.2f}%')

# 10. Error Analysis (optional but helpful)
incorrect_images = []
incorrect_labels = []
predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                incorrect_images.append(images[i].cpu())
                incorrect_labels.append(labels[i].cpu())
                predictions.append(predicted[i].cpu())

# Displaying the first 5 incorrect predictions
fig, axes = plt.subplots(1, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flatten()):
    if i < len(incorrect_images):
        img = incorrect_images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {incorrect_labels[i]} - Pred: {predictions[i]}')
        ax.axis('off')
plt.show()
