# Pneumonia Detection using EfficientNetB0

## 🩺 Objective

The goal of this project is to build a robust and efficient deep learning model for **binary classification** of chest X-ray images into `PNEUMONIA` and `NORMAL` categories using **EfficientNetB0** as the base model. The solution focuses on:
- Achieving high precision and recall due to the critical nature of medical diagnosis.
- Preventing overfitting while leveraging transfer learning.
- Improving model performance through fine-tuning and training strategies.

---

## 📁 Dataset

- Balanced dataset of chest X-ray images categorized into `NORMAL` and `PNEUMONIA`.
- Split: 80% training / 20% validation
- Preprocessed to 224x224 resolution

---

## 🛠️ Model Versions & Techniques Used

### ✅ Version 1 – **Baseline EfficientNetB0**
- **Base Model**: EfficientNetB0 with pretrained ImageNet weights.
- **Frozen Base**: All layers frozen initially.
- **Added Layers**: GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid).
- **Optimizer**: Adam (lr = 1e-4)
- **Loss**: Binary Crossentropy
- **Data Augmentation**: Applied during training.

#### 📊 Evaluation:
- **Accuracy**: `93.12%`
- **Precision**: `0.92 (NORMAL)`, `0.95 (PNEUMONIA)`
- **Recall**: `0.95 (NORMAL)`, `0.91 (PNEUMONIA)`
- **F1-Score**: `0.93`
- **AUC**: ~0.98
- **Confusion Matrix**: `[[260, 33], [16, 277]]`

#### 📌 Insight:
- Model performed well, but showed signs of imbalance between precision and recall.
- Room for improvement by allowing selective layer training in the base model.

---

### ✅ Version 2 – **Fine-Tuned EfficientNetB0 (Partial Unfreeze)**
- **Unfreezing Strategy**: Top 20 layers of the base model unfrozen.
- **Re-compile with**: Adam (lr = 1e-5) for fine-tuning.
- **Training Strategy**: Freeze → Train → Unfreeze → Retrain (5 epochs).
- **Data Augmentation**: Maintained for generalization.

#### 📊 Evaluation:
- **Accuracy**: `93.86%`
- **Precision**: `0.93 (NORMAL)`, `0.95 (PNEUMONIA)`
- **Recall**: `0.95 (NORMAL)`, `0.92 (PNEUMONIA)`
- **F1-Score**: `0.94`
- **AUC**: `0.9845`
- **Confusion Matrix**: `[[279, 14], [22, 271]]`

#### 📌 Insight:
- Unfreezing select top layers allowed the model to learn more specialized features.
- Improved recall for NORMAL cases and balanced overall metrics.
- High AUC indicates excellent class separability.

---

### ✅ Version 3 – **Extended Fine-Tuning (Top 20 Layers + ReduceLROnPlateau)**
- **Unfreezing Strategy**: Top 30 layers of EfficientNetB0 unfrozen.
- **Learning Rate Adjustment**: Used `ReduceLROnPlateau` to adapt learning.
- **Optimizer**: Adam (lr = 1e-5 after unfreezing).
- **Training Strategy**: 6 epochs frozen + 10 epochs fine-tuning.
- **Data Augmentation**: Used aggressively.

#### 📊 Evaluation:
- **Accuracy**: `94.71%`
- **Precision**: `0.94 (NORMAL)`, `0.95 (PNEUMONIA)`
- **Recall**: `0.96 (NORMAL)`, `0.94 (PNEUMONIA)`
- **F1-Score**: `0.95`
- **AUC**: `0.9874`
- **Confusion Matrix**: `[[280, 13], [18, 275]]`

#### 📌 Insight:
- Best version so far — very well-balanced metrics.
- Minimal overfitting and excellent generalization.
- Strong candidate for deployment or real-world validation.

---

## 🔚 Final Comparison Table

| Version          | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score | AUC    |
|------------------|----------|------------------|--------------|----------|--------|
| Baseline v1      | 93.12%   | 0.935             | 0.93         | 0.93     | ~0.980 |
| Fine-Tuned v2    | 93.86%   | 0.94              | 0.935        | 0.94     | 0.9845 |
| Extended v3      | 94.71%   | 0.945             | 0.95         | 0.95     | 0.9874 |

---


