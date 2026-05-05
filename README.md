 🩺 Chest X-Ray Classification — Pneumonia Detection

This project implements a complete **machine learning pipeline** for detecting pneumonia from chest X-ray images using classical ML algorithms. It also includes a **Flask-based web dashboard** for training models and visualizing results in real time.

---

## 📖 Overview

The goal of this project is to classify chest X-ray images into:
- **NORMAL**
- **PNEUMONIA**

We compare three machine learning models:
- K-Nearest Neighbours (KNN)
- Decision Tree
- Gaussian Naive Bayes

The system includes preprocessing, training, evaluation, and deployment via a web interface.

---

## 🎯 Features

- ✅ Image preprocessing pipeline (grayscale, resize, normalization)
- ✅ Training of multiple ML models
- ✅ Performance comparison (Accuracy, Precision, Recall, F1-score)
- ✅ Confusion matrix visualization
- ✅ Flask web dashboard for:
  - Training models
  - Monitoring progress
  - Viewing results
- ✅ Model saving using joblib

---

## 📂 Dataset

- **Dataset:** Chest X-Ray Images (Pneumonia)
- Total images: ~5,800+
- Classes:
  - NORMAL
  - PNEUMONIA

---

## ⚙️ Preprocessing Steps

- Convert images to grayscale
- Resize to **64 × 64**
- Flatten into 1D feature vectors
- Normalize pixel values (0–1)
- Encode labels:
  - NORMAL → 0
  - PNEUMONIA → 1

---

## 🤖 Models Used

### 1. K-Nearest Neighbours (KNN)
- k = 5
- Highest accuracy (~74.5%)
- Strong pneumonia detection

### 2. Decision Tree
- max_depth = 20
- Interpretable model
- Moderate performance

### 3. Gaussian Naive Bayes
- Fastest training
- Balanced performance across classes

---

## 📊 Results Summary

| Model | Accuracy | F1 Score | Training Time |
|------|---------|---------|--------------|
| KNN | ~74.5% | ~70.4% | 1.8s |
| Decision Tree | ~74.0% | ~70.9% | 24.8s |
| Naive Bayes | ~72.0% | ~71.9% | 0.26s |

---

## 🌐 Web Application

Built with Flask to provide an interactive interface.

### Routes:
- `/` → Home page
- `/train` → Train models
- `/status` → Check training progress
- `/results` → View results

### Features:
- Real-time training updates
- Confusion matrix visualization
- Model comparison dashboard

---

## 🛠️ Tech Stack

- Python 3.12
- Flask
- Scikit-learn
- NumPy
- Pillow
- Matplotlib & Seaborn
- Joblib

---

## 📁 Project Structure

