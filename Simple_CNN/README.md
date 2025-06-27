# 🩺 Pneumonia Detection from Chest X-Rays using Simple CNN

## 📌 Objective

The goal of this project is to build a lightweight, interpretable Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. We aim to:

- Achieve strong predictive performance with a custom model
- Understand and tackle overfitting through experimentation
- Compare multiple training strategies and observe their effects on generalization

---

## Model Architecture (Common Base)


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

---


## 🔬 Experimental Versions

### 🔁 Version 1: Baseline Simple CNN
**✅ Techniques**  
- 2 Conv layers  
- Dropout (0.3)  
- Binary Crossentropy loss  
- Adam optimizer  

**📊 Results**  
- Accuracy: 91.64%  
- AUC: 0.9800  
- Confusion Matrix:  

**📌 Insights**  
- High training accuracy (99.8%) vs validation (93%) → overfitting  
- Pneumonia recall: 0.95  
- Precision affected for normal class  

---

### 🔁 Version 2: Data Augmentation
**✅ Techniques**  
- Data augmentation (flip/rotation/zoom/contrast)  
- Dropout (0.3)  

**📊 Results**  
- Accuracy: 92.49%  
- AUC: 0.9789  
- Confusion Matrix:  

**📌 Insights**  
- Improved generalization (higher AUC/recall)  
- Reduced but persistent overfitting  
- Insufficient regularization  

---

### 🔁 Version 3: Data Augmentation + L2 + EarlyStopping
**✅ Techniques**  
- Data augmentation  
- L2 regularization (λ=0.01)  
- EarlyStopping on val_loss  
- Dropout (0.3)  

**📊 Results**  
- Accuracy: 94.20%  
- AUC: 0.9848  
- Class Metrics:  
| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Normal      | 0.95      | 0.94   | 0.94     |
| Pneumonia   | 0.94      | 0.95   | 0.94     |  
- Confusion Matrix:  

**📌 Insights**  
- Balanced performance across classes  
- Eliminated overfitting  
- Clean learning curves with early convergence  

---

## 📊 Version Comparison
| Version             | Accuracy | AUC    | Overfitting | Key Improvement          |
|---------------------|----------|--------|-------------|--------------------------|
| Baseline           | 91.64%   | 0.9800 | High        | Initial model           |
| + Data Augmentation| 92.49%   | 0.9789 | Moderate    | Better generalization   |
| + L2 + EarlyStop   | **94.20%** | **0.9848** | **None** | Optimal regularization |

## 📌 Final Takeaways
1. **Regularization is critical** - L2 regularization and early stopping doubled accuracy gains from data augmentation alone  
2. **Interpretability** - Simple CNN achieves competitive performance (94.2% accuracy) vs complex architectures  
3. **Clinical relevance** - Balanced precision/recall (94-95%) reduces misdiagnosis risks  
4. **Training strategy** - Progressive regularization yielded:  
 - 2.56% absolute accuracy improvement  
 - Elimination of overfitting  
 - 0.0048 AUC increase over baseline  


