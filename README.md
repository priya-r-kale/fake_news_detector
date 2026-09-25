# 📰 Fake News Detector

## 📌 Project Overview

The Fake News Detector is a Machine Learning project that predicts whether a news article is **Real** or **Fake**.

Users can paste a news headline or article into the application and click the **Analyze** button to get the prediction.

---

## 🚀 Features

- Detects Fake and Real news
- Simple and easy-to-use interface
- Shows prediction confidence
- Fast prediction using Machine Learning
- Built with Streamlit

---

## 📊 Dataset Used

This project uses the **Fake and Real News Dataset**.

- Fake.csv → 23,481 articles
- True.csv → 21,417 articles
- Total Dataset → 44,898 articles

---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib

---

## 📁 Project Structure

```
fake_news_detector/
│
├── app.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
└── model/
    ├── model.joblib
    └── vectorizer.joblib
```

---

## ⚙ Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## ▶ Train the Model

```bash
python train.py
```

---

## 🌐 Run the Application

```bash
streamlit run app.py
```

---

## 💻 How to Use

1. Open the application.
2. Paste a news headline or article.
3. Click **Analyze**.
4. View whether the news is **REAL** or **FAKE**.

---

## 📈 Future Improvements

- Improve prediction accuracy
- Add Deep Learning models
- Detect news from URLs
- Better user interface
- Support multiple languages

---

## 👨‍💻 Developer

**Omkar Kathare**

📧 Email: pkale3922@gmail.com

---

## 🙏 Thank You

Thank you for visiting this project.
I hope this project helps you understand Machine Learning and Fake News Detection.
