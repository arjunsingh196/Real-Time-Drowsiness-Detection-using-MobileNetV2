# 👁️ Real-Time Drowsiness Detection using MobileNetV2

> A real-time drowsiness detection system using MobileNetV2 and PyTorch. Classifies eye states (open vs. closed) from live webcam feed using Haar cascade and triggers an audio alert when drowsiness is detected. Achieves 90%+ accuracy with efficient preprocessing, transfer learning, and real-time inference.

---

## 🚀 Overview

This project detects drowsiness by analyzing eye images in real-time via webcam. It classifies eye states using a deep learning model and triggers an audio alert if the eyes remain closed — useful in applications like driver monitoring systems or fatigue detection setups.

---

## 🔍 Features

- 🧠 **Transfer Learning** using pretrained **MobileNetV2**
- 📊 Achieves **90%+ real-time accuracy**
- 🎯 Binary classification: Open vs Closed Eyes
- 📷 **Haar Cascade** for real-time eye region detection
- 🔔 **Beep alert system** for drowsiness warnings
- ⚡ Fast preprocessing with `pickle` caching
- 💻 Real-time performance on CPU or GPU (CUDA)

---

## 🧱 Model Architecture

- **Backbone**: `torchvision.models.mobilenet_v2(pretrained=True)`
- **Modified Classifier**:
  ```python
  nn.Sequential(
      nn.Flatten(),
      nn.Linear(1280, 1),
      nn.Sigmoid()
  )
