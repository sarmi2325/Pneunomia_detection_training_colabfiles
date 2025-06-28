# Pneumonia Detection with MobileNetV2

## 🚀 Objective

To build an efficient and accurate deep learning model using **MobileNetV2** for classifying chest X-ray images into **NORMAL** and **PNEUMONIA** categories, and to evaluate the effect of fine-tuning (unfreezing top layers) on the model's performance.

---

## 🔧 Version 1: MobileNetV2 (Frozen Base)

### ⚖️ Techniques Used:

* **Transfer Learning** with pre-trained ImageNet weights
* **Frozen base model**: Entire MobileNetV2 base kept non-trainable
* **Global Average Pooling** + **Dropout** + Dense Output
* **Binary Crossentropy Loss**
* **Adam Optimizer**
* **EarlyStopping Callback** to avoid overfitting

### 📊 Evaluation Metrics:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 87.20% |
| Precision | 0.87   |
| Recall    | 0.87   |
| F1-Score  | 0.87   |
| AUC Score | 0.9256 |

### 🔍 Insights:

* Model trained fast but plateaued early.
* Performance was good, but **underfitting** observed.
* Not enough learning capacity due to fully frozen layers.

---

## 🔧 Version 2: MobileNetV2 (Fine-Tuned)

### ⚖️ Techniques Used:

* **Unfrozen top 30 layers** of the base model
* **Lower learning rate** during fine-tuning
* **Data Augmentation** (rotation, flipping, etc.)
* **Dropout**, **Batch Normalization**
* **EarlyStopping** and **AUC Monitoring**

### 📊 Evaluation Metrics:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 96.59% |
| Precision | 0.95   |
| Recall    | 0.96   |
| F1-Score  | 0.95   |
| AUC Score | 0.9925 |

### 🔍 Insights:

* Fine-tuning drastically improved performance.
* Model learned better discriminative features.
* Generalized well to validation set (no overfitting).
* Final model is robust and suitable for deployment.

---

## 🔢 Final Comparison

| Model Version            | Accuracy | Precision | Recall | F1 Score | AUC    |
| ------------------------ | -------- | --------- | ------ | -------- | ------ |
| MobileNetV2 (Frozen)     | 87.20%   | 0.87      | 0.87   | 0.87     | 0.9256 |
| MobileNetV2 (Fine-tuned) | 96.59%   | 0.95      | 0.96   | 0.95     | 0.9925 |

---

## 📅 Conclusion

* **Freezing** helped with fast convergence, but limited learning.
* **Fine-tuning top layers** gave significant performance boost.
* MobileNetV2 is a strong baseline for medical imaging tasks, especially when paired with good data augmentation and careful unfreezing.

> Next steps: Consider adding interpretability tools (like Grad-CAM) or optimizing for real-time deployment.

---
