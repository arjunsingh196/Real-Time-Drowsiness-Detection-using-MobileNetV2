# %%
# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from PIL import Image
import winsound
import time

# %%
# Load a single grayscale image and display it
img_path = r"D:\PROJECTS\HandWrittenDigitRecognition\New folder\archive\mrleyedataset\Train_data_SET\Closed_Eyes\s0001_00002_0_0_0_0_0_01.png"
img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

print("Image shape:", img_array.shape)

# %%
# Load and visualize an image from the dataset directory
Datadirectory = r"D:\PROJECTS\HandWrittenDigitRecognition\New folder\archive\mrleyedataset"
Classes = [
    r"Train_data_SET\Closed_Eyes",
    r"Train_data_SET\Open_Eyes"
]

# Display the first image from each class
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        plt.imshow(img_array, cmap='gray')
        plt.title(f"Class: {os.path.basename(category)}")
        plt.axis("off")
        plt.show()
        break
    break

# %%
# Parameters and Paths
img_size = 224
Datadirectory = r"D:\PROJECTS\HandWrittenDigitRecognition\New folder\archive\mrleyedataset"
Classes = [
    r"Train_data_SET\Closed_Eyes",
    r"Train_data_SET\Open_Eyes"
]

# %%
# Resize and display a sample image
img_path = os.path.join(Datadirectory, Classes[0], "s0001_00002_0_0_0_0_0_01.png")
img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
new_array = cv2.resize(backtorgb, (img_size, img_size))

plt.imshow(new_array)
plt.title("Resized RGB Image")
plt.axis("off")
plt.show()

# %%
# Read all images and store them with labels
training_data = []

def create_training_data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(f"Total training samples: {len(training_data)}")

# %%
# Shuffle and split data
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X / 255.0
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# %%
# Save the processed data using pickle
with open("X.pickle", "wb") as pickle_out:
    pickle.dump(X, pickle_out)

with open("y.pickle", "wb") as pickle_out:
    pickle.dump(y, pickle_out)

# %%
# Load the pickled data
with open("X.pickle", "rb") as f:
    X = pickle.load(f)

with open("y.pickle", "rb") as f:
    y = pickle.load(f)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# %%
# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# %%
# Load MobileNetV2 and freeze base
mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
for param in mobilenet.features.parameters():
    param.requires_grad = False

mobilenet.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1280, 1),
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet = mobilenet.to(device)

# %%
# Loss, optimizer, and training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(mobilenet.parameters(), lr=0.001)

# %%
# Training loop
epochs = 1
mobilenet.train()

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = mobilenet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {correct/total:.4f}")

# %%
# Save the trained model
torch.save(mobilenet.state_dict(), "mobilenet_drowsiness.pth")

# %%
# Load model for inference
mobilenet_infer = torchvision.models.mobilenet_v2(pretrained=False)
mobilenet_infer.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1280, 1),
    nn.Sigmoid()
)
mobilenet_infer.load_state_dict(torch.load("mobilenet_drowsiness.pth"))
mobilenet_infer.eval()
mobilenet_infer.to(device)

# %%
# Live webcam detection with Haar cascade, sound alert, and multi-eye detection

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def live_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("🟢 Live Drowsiness Detection Started — Press 'q' to quit")
    frame_skip = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        closed_eye_detected = False

        for (ex, ey, ew, eh) in eyes:
            eye_img = gray[ey:ey+eh, ex:ex+ew]
            eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(eye_rgb, (224, 224))
            img_tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                output = mobilenet_infer(img_tensor)
                prediction = output.item() > 0.5
                label = "Open Eyes" if prediction else "Closed Eyes"
                color = (0, 255, 0) if prediction else (0, 0, 255)

                if not prediction:
                    closed_eye_detected = True

            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(frame, label, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if closed_eye_detected:
            winsound.Beep(2000, 300)  # 2000 Hz — sharper beep

        cv2.imshow("Live Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# %%
# Run live detection
live_detection()

# %%
