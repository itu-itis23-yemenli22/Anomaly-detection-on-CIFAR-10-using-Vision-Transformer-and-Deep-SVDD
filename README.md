# 🔎 Anomaly Detection via Deep One-Class Classification Using Vision Transformers

This project presents a one-class anomaly detection approach using a pretrained Vision Transformer (ViT) and Deep SVDD. The goal is to detect anomalies in the CIFAR-10 dataset by learning only the distribution of the "ship" class and identifying deviations.

---

## 📌 Overview

- **Problem**: Most real-world anomaly detection tasks lack anomalous samples during training.
- **Solution**: Learn the latent space of the normal class (ship) using ViT and detect anomalies using Deep SVDD.
- **Comparison**: Classical anomaly detection methods (PCA, One-Class SVM, Isolation Forest) were applied on the same ViT feature embeddings.

---

## 🧠 Methods

### 🔹 Feature Extraction

- Model: Pretrained Vision Transformer (ViT-Base)
- Dataset: CIFAR-10
- Input: Normalized and resized images
- Feature vector: Final hidden representation of the CLS token

### 🔹 Anomaly Detection Models

- **Deep SVDD**  
  Minimizes the volume of a hypersphere enclosing the normal class representations  
- **PCA**  
  Reconstruction error-based anomaly scoring  
- **One-Class SVM**  
  Boundary learning in kernel space  
- **Isolation Forest**  
  Anomaly isolation by recursive partitioning

---

## 📊 Results

| Method            | AUC    | F1 Score |
|-------------------|--------|----------|
| Learned Deep SVDD | 0.9888 | 0.9813   |
| PCA               | 0.9454 | 0.9803   |
| One-Class SVM     | 0.8866 | 0.9652   |
| Isolation Forest  | 0.8693 | 0.9496   |
| Basic Deep SVDD   | 0.5272 | 0.1050   |

- Deep SVDD outperformed all other methods in both AUC and classification metrics.
- PCA was a strong classical baseline.
- Isolation Forest and One-Class SVM performed moderately.

---

## 📁 Files

- `BLG454codesipynb.ipynb`: End-to-end implementation including feature extraction, model training, and evaluation
- `7906605.pdf`: Final project report with detailed methodology and figures

---

## 📚 Dataset

- CIFAR-10  
- Normal class: `ship`  
- Anomalous classes: `airplane`, `cat`, `dog`, `truck`, etc.  
- [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 👥 Authors

- Berker Eriş  
- Enes Yemenli  
- Emircan Sarı  

> Department of Artificial Intelligence and Data Engineering, Istanbul Technical University  
> Contact: erisb22@itu.edu.tr
> Contact: yemenli22@itu.edu.tr
>  Contact: sari22@itu.edu.tr
---

## 📌 Notes

This project was completed as part of the **BLG454E - Deep Learning** course in Spring 2025.  
The methods were implemented using PyTorch and Scikit-learn.

---

