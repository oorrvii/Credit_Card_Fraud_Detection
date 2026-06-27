# 🛡️ FraudShield — Credit Card Fraud Detector

A machine learning web app that classifies credit card transactions as **Legitimate** or **Fraudulent** in real time using a Random Forest model.

🔗 **Live Demo:** [creditcardfrauddetection-q8frju3uzky7dgbemkewsd.streamlit.app](https://creditcardfrauddetection-q8frju3uzky7dgbemkewsd.streamlit.app/)

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest |
| Precision (Fraud) | 93% |
| Recall (Fraud) | 79% |
| F1 Score (Fraud) | 0.86 |
| Class Imbalance Handling | `class_weight='balanced'` |

---

## 🗂️ Dataset

**ULB Credit Card Fraud Detection Dataset** — Kaggle

| Stat | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Features | V1–V28 (PCA), Amount, Time |
| Source | ULB Machine Learning Group |

The dataset is highly imbalanced — only 0.17% of transactions are fraudulent. `class_weight='balanced'` was used to handle this without oversampling.

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn** — Random Forest, StandardScaler
- **Streamlit** — Web app & deployment
- **NumPy / Pandas** — Data handling
- **Pickle** — Model serialization

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py                  # Streamlit web app
├── fraud_classifier.pkl    # Trained Random Forest model
├── scaler.pkl              # Fitted StandardScaler
├── requirements.txt        # Dependencies
└── README.md
```

---

## 📌 How It Works

1. Select a sample transaction (3 legitimate, 3 suspicious)
2. Features are scaled using the fitted StandardScaler
3. Random Forest model classifies the transaction
4. Result displayed instantly — ✅ Legitimate or 🚨 Fraudulent

---

## 💡 Key Learnings

- Handling **severely imbalanced datasets** using `class_weight='balanced'`
- Why **precision and recall matter more than accuracy** for fraud detection
- Difference between **False Positives** (blocking innocent users) vs **False Negatives** (missing fraud)
- End-to-end ML pipeline from raw data → model → deployed web app

---

## ⚠️ Disclaimer

This tool is built for **educational and portfolio purposes only** and is not intended for production use in real financial systems.

---

## 👩‍💻 Author

Built by **Oorvi** as part of an ML learning journey.
