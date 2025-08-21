
# 🩺 Multi-task Breast Ultrasound Analysis

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths among women.
Ultrasound imaging provides a **non-invasive and affordable** method for early detection.

This project presents two deep learning approaches for analyzing the **BUSI Breast Ultrasound Dataset**:

1. **Two-Stage Pipeline**

   * **U-Net** for segmentation → localize tumors
   * **ResNet50** for classification → predict **Normal, Benign, or Malignant**

2. **Multi-task Model**

   * A **shared ResNet50 backbone** with

     * Segmentation head (decoder)
     * Classification head (fully connected)
   * Trained end-to-end with a **weighted loss**

Both approaches are implemented in **PyTorch** (on Kaggle Notebook).

---

## 📂 Dataset: BUSI

* Source: [Breast Ultrasound Images (BUSI) on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
* Classes: **Normal, Benign, Malignant**
* Total images: **1578**
* Ground truth masks: **1578**
* Format: Grayscale ultrasound images with corresponding segmentation masks

**Folder Structure:**

```
Dataset_BUSI_with_GT/
│── normal/
│── benign/
│── malignant/
```

---

## 🔄 Workflow

### 1️⃣ Two-Stage Pipeline

```
[Input Image] 
      ↓
U-Net (Segmentation) → Tumor Mask
      ↓
Crop Tumor ROI
      ↓
ResNet50 (Classification)
      ↓
Predicted Label: Normal / Benign / Malignant
```

---

### 2️⃣ Multi-task Model

```
[Input: 224x224x3 Image]
      ↓
Shared Backbone: ResNet50 (pretrained)
      ↓                  ↓
Segmentation Head        Classification Head
(Upsample → Conv)        (FC Layer → Softmax)
      ↓                  ↓
Predicted Mask           Predicted Label
```

📌 **Loss Function**

* Segmentation Loss: Binary Cross Entropy (BCE)
* Classification Loss: CrossEntropyLoss
* Total Loss: `loss = loss_seg + loss_cls`

---

## 📝 Flowchart

![Flowchart](docs/flowchart.png)

---

## ⚙️ Implementation

All experiments were implemented in **Kaggle Notebook**.
Code is modularized into:

* `unet.py` → U-Net model
* `resnet_classifier.py` → ResNet50 classifier
* `multitask_model.py` → Joint segmentation + classification
* `dataset.py` → Dataset & preprocessing
* `train.py` → Training loop
* `inference.py` → Run inference on new images
* `utils.py` → Metrics + visualization

---

## 📊 Results

### 🔹 Segmentation (U-Net)

* Metric: Dice Score, IoU
* Example masks:
  *(Add example visualizations here)*

### 🔹 Classification (ResNet50)

* Accuracy: XX%
* Confusion Matrix:
  *(Insert confusion matrix plot here)*

### 🔹 Multi-task Model

* Accuracy: XX%
* Segmentation Dice: XX%
* Joint Training → More efficient + clinical decision support ready

---

## 🚀 Usage

### 1. Clone Repo

```bash
git clone https://github.com/username/breast-ultrasound-analysis.git
cd breast-ultrasound-analysis
```

### 2. Install Dependencies

```bash
conda env create -f environment.yml
conda activate busi-env
```

or

```bash
pip install -r requirements.txt
```

### 3. Run Training

```bash
python src/train.py --model multitask --epochs 50
```

### 4. Inference Demo

```bash
python src/inference.py --image sample.png --model results/model_best.pth
```

---

## 📈 Visualizations

* Training vs Validation Loss
* Accuracy Curves
* Example Segmentation Masks + Predicted Labels

*(Add plots + screenshots here for attractiveness)*

---

## 🛠️ Future Work

* Attention U-Net for improved segmentation
* Hyperparameter tuning
* Extending dataset with data augmentation
* Deploy as a web-based clinical decision support tool

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

