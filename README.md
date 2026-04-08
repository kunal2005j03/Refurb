# 📱 PhoneGuard AI — Smartphone Damage Detector

A complete end-to-end image classification system that detects whether a
smartphone is **damaged** or **not damaged**, built with TensorFlow (MobileNetV2)
and a polished Streamlit web interface.

---

## 📁 Project Structure

```
smartphone_classifier/
├── dataset/
│   ├── damaged/          ← place damaged phone images here
│   └── not_damaged/      ← place undamaged phone images here
├── model/
│   └── model.h5          ← saved after training
├── train.py              ← model training script
├── predict.py            ← inference module (importable + CLI)
├── app.py                ← Streamlit web application
├── setup_dataset.py      ← generates synthetic demo images
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start (with synthetic demo data)

```bash
# Step 1 — Generate demo images (skip if you have real photos)
python setup_dataset.py

# Step 2 — Train the model
python train.py

# Step 3 — Launch the web app
streamlit run app.py
```

---

## 🖼️ Using Your Own Dataset

Place your images inside the `dataset/` folder:

```
dataset/
├── damaged/        ← JPG / PNG / WEBP photos of damaged phones
└── not_damaged/    ← JPG / PNG / WEBP photos of intact phones
```

Recommended: **100–500 images per class** for reasonable accuracy.

Then run:

```bash
python train.py
streamlit run app.py
```

---

## 🧠 Model Architecture

| Layer                  | Details                        |
|------------------------|--------------------------------|
| **Base**               | MobileNetV2 (ImageNet weights, frozen) |
| GlobalAveragePooling2D | —                              |
| Dense                  | 128 units, ReLU                |
| Dropout                | 0.5                            |
| **Output**             | 2 units, Softmax               |

- **Optimizer** : Adam (lr = 1e-4)
- **Loss**      : Categorical Cross-Entropy
- **Input size**: 224 × 224 × 3

---

## 🔍 CLI Prediction

You can also run predictions from the terminal:

```bash
python predict.py path/to/phone_image.jpg
```

Output example:
```
📋  Result:
    Label      : damaged
    Confidence : 94.32%
    All scores : {'damaged': 0.9432, 'not_damaged': 0.0568}

    Final Status: ⚠️  Damaged
```

---

## 📊 Data Augmentation

Applied during training to reduce overfitting:

| Technique         | Value  |
|-------------------|--------|
| Rotation          | ±20°   |
| Zoom              | 15%    |
| Horizontal flip   | Yes    |
| Width shift       | 10%    |
| Height shift      | 10%    |
| Shear             | 10%    |

---

## 🌐 Streamlit App Features

- 📎 Drag-and-drop image upload (JPG, PNG, WEBP)
- 🖼️ Live image preview
- 🧠 One-click inference
- 📊 Score breakdown with progress bars
- ✅ / ⚠️ Verdict: **Good Condition** or **Damaged**
