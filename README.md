# 🍅 Tomato Disease Prediction Model

This repository contains a machine learning model built with neural networks to detect and classify **tomato plant diseases** from leaf images. The model predicts whether a given tomato plant image shows signs of:

- **Early Blight**
- **Late Blight**
- **Healthy**

> 🚧 Note: This project currently includes only the **trained model**. Integration, deployment, and UI work are yet to be done.

---

## 🧠 Model Overview

This project uses a **convolutional neural network (CNN)** to classify tomato plant images into one of the three classes. It has been trained on a labeled dataset of tomato leaf images with annotated disease categories.

### Classes:
- `Early Blight`
- `Late Blight`
- `Healthy`

---

## 📁 Project Structure

```
tomato-disease-predictor/
├── model/
│   └── tomato_disease_model.h5          # Trained model file
├── notebooks/
│   └── training_and_evaluation.ipynb    # Model training and evaluation notebook
├── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/prashantpoudel745/tomato_disease_classification
cd tomato-disease-classification
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy opencv-python
```

### 3. Predict an Image

```python
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model('model/tomato_disease_model.h5')

# Load and preprocess image
img = cv2.imread('path_to_image.jpg')
img = cv2.resize(img, (224, 224))          # Resize image to model input size
img = img / 255.0                          # Normalize pixel values
img = np.expand_dims(img, axis=0)         # Add batch dimension

# Predict
prediction = model.predict(img)
classes = ['Early Blight', 'Late Blight', 'Healthy']
print("Predicted Class:", classes[np.argmax(prediction)])
```

---

## 🧪 Model Performance

Detailed training metrics including accuracy and loss curves can be found in:

```
notebooks/training_and_evaluation.ipynb
```

---

## 📌 Next Steps

- [ ] Add image upload UI (web or mobile)
- [ ] Build REST API using FastAPI or Flask
- [ ] Deploy to cloud (e.g., Render, Vercel, AWS)
- [ ] Integrate real-time predictions via camera input

---

## 🙏 Acknowledgements

Model inspired by the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).

---

## 📜 License

This project is licensed under the **MIT License**.
