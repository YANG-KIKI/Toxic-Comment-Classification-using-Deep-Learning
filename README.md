# 💬 Toxic Comment Classification with Fairness-Aware Deep Learning Models

This repository contains our final project for the **Deep Learning** course at DSBA. Our group participated in the **Kaggle Toxic Comment Classification Challenge**, where the goal is to automatically detect toxic content in online comments, with a strong emphasis on **fairness across demographic subgroups**.

## 🧠 Project Overview

As online platforms expand, it becomes increasingly vital to automatically filter out toxic content such as insults, threats, and identity attacks. However, many models introduce bias against minority groups. Our objective is to build robust NLP models that not only identify toxicity but also **ensure fairness across identity subgroups** using the **Worst-Group Accuracy (WGA)** metric.

We explored and compared three different models:
- Logistic Regression (baseline)
- LSTM (Recurrent Neural Network)
- BERT (Transformer-based)

## 📁 Repository Structure

```bash
├── EDA.ipynb                  # Data cleaning, distribution analysis, word clouds, correlation
├── Logistics_Regression.ipynb # TF-IDF + Logistic Regression model with fairness metrics
├── LSTM.ipynb                 # LSTM model with text tokenization, embedding, and group-wise evaluation
├── BERT_1&2&3_DL.ipynb        # BERT modeling and experiments (baseline, save-by-WGA, dynamic loss)
└── README.md                  # This file
```

## 📊 Dataset

The data comes from the Kaggle competition **[Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)**. It includes:

- `train_x` / `val_x` / `test_x`: User comments  
- `train_y` / `val_y`: Toxicity labels (`y`) and identity features (e.g., `male`, `female`, `black`, `muslim`, etc.)  
- **Demographic groups**: 8 identities  
- **Toxicity types**: 6 features, such as `insult`, `obscene`, `threat`, etc.

---

## 🧪 Models & Evaluation

### 🔹 Logistic Regression
- **Preprocessing**: TF-IDF + demographic/toxicity feature standardization  
- **Training**: BCEWithLogitsLoss, evaluated using Worst-Group Accuracy (WGA)  
- **Results**:
  - WGA (Validation): `0.9121`
  - WGA (Test): `0.6811`

### 🔹 LSTM
- **Text Preprocessing**: Tokenized and padded sequences (max length = 100)  
- **Architecture**: 2-layer LSTM with 128 hidden units + dropout  
- **Results**:
  - WGA (Validation): `0.7318`
  - WGA (Test): `0.7524`

### 🔹 BERT (3 Variants)
- **Base Model**: `bert-base-uncased`  
- **Loss Design**: Added weighted loss to reduce demographic bias  

#### 🔸 BERT_1
- Vanilla loss + save by `val_loss`  
- **WGA (Test)**: `0.7935`

#### 🔸 BERT_2
- Save model based on `WGA` instead of `val_loss`  
- **WGA (Test)**: `0.7861`

#### 🔸 BERT_3
- Dynamic group-weighted loss function + save by WGA  
- **WGA (Test)**: `0.7171`

✅ **Final Model**: `BERT_1` — best trade-off between performance and fairness.

---

## 📈 Evaluation Metric

We use **Worst-Group Accuracy (WGA)**:  
> The minimum classification accuracy across all identity subgroups  
> (e.g., `female-toxic`, `muslim-non-toxic`, etc.)

This metric helps us understand **model fairness**, not just overall performance.

---

## 📌 Key Highlights

- Tackled dataset imbalance using:
  - Oversampling
  - Weighted loss functions
  - Threshold tuning  
- Designed **fairness-aware loss functions** and **model-saving strategies**  
- Demonstrated that **BERT outperforms traditional models** in capturing context and reducing bias
