

# 🩺 Multi-task Breast Ultrasound Analysis

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths among women.  
Ultrasound imaging provides a **non-invasive and affordable** method for early detection.

This project presents two deep learning approaches for analyzing the **BUSI Breast Ultrasound Dataset**:

###  Approach 1: Two-Stage Pipeline

- **U-Net** for segmentation → localize tumors
- **ResNet50** for classification → predict **Normal, Benign, or Malignant**

###  Approach 2: Multi-task Model

- A **shared ResNet50 backbone** with:
  - Segmentation head (decoder)
  - Classification head (fully connected)
- Trained end-to-end with a **weighted loss**

Both approaches are implemented in **PyTorch** (on Kaggle Notebooks).

---

## 📂 Dataset: BUSI

* Source: [Breast Ultrasound Images (BUSI) on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
* Classes: **Normal, Benign, Malignant**
* Total images: **1578**
* Ground truth masks: **1578**
* Format: Grayscale ultrasound images with corresponding segmentation masks

**Folder Structure:**

```

Dataset\_BUSI\_with\_GT/
│── normal/
│── benign/
│── malignant/

```

---

## 🔄 Workflow

### 1️⃣ Two-Stage Pipeline

```

\[Input Image]
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

\[Input: 224x224x3 Image]
↓
Shared Backbone: ResNet50 (pretrained)
↓                  ↓
Segmentation Head        Classification Head
(Upsample → Conv)        (FC Layer → Softmax)
↓                  ↓
Predicted Mask           Predicted Label

````

📌 **Loss Function**

* Segmentation Loss: Binary Cross Entropy (BCE)
* Classification Loss: CrossEntropyLoss
* Total Loss: `loss = loss_seg + loss_cls`

---

## 📝 Flowchart

![Flowchart](docs/flowchart.png)

---

## ⚙️ Implementation

All experiments were implemented in **Kaggle Notebooks** (also included in the `notebooks/` folder):

| Notebook                          | Description                                  |
|-----------------------------------|----------------------------------------------|
| `1_segmentation_unet.ipynb`       | U-Net segmentation training + evaluation     |
| `2_classification_resnet50.ipynb` | ResNet50 classification on tumor crops       |
| `3_multitask_model.ipynb`         | Joint segmentation + classification model    |

👉 You can open these notebooks directly in Kaggle or Jupyter Notebook to view code, results, and visualizations.

---

## 📊 Results

### 🔹 Segmentation (U-Net)

* Metrics: Dice Score, IoU
* Visual Examples: *(Add sample mask overlays in `results/` folder)*

### 🔹 Classification (ResNet50)

* Accuracy: **XX%**
* Confusion Matrix: *(Insert confusion matrix plot)*

### 🔹 Multi-task Model

* Classification Accuracy: **XX%**
* Segmentation Dice Score: **XX%**
* Joint Training = more efficient + better clinical relevance

---

## 📈 Visualizations

Include the following plots (place in `results/`):

* Training vs Validation Loss
* Accuracy Curves
* Confusion Matrix
* Example Segmentation Masks with Predicted Labels

---

## 🚀 Usage

### 1. Clone Repo

```bash
git clone https://github.com/username/breast-ultrasound-analysis.git
cd breast-ultrasound-analysis
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Open Notebooks

You can open any notebook in Jupyter or VSCode:

```bash
jupyter notebook notebooks/1_segmentation_unet.ipynb
```

Or upload the notebook to [Kaggle](https://www.kaggle.com/) and run there.

---

## 🛠️ Future Work

* Attention U-Net for improved segmentation
* Data augmentation to improve generalization
* Hyperparameter tuning
* Deploy as a web-based clinical decision support tool

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

````

---


