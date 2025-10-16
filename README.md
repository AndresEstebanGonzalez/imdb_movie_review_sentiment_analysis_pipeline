# 🎬 IMDB Sentiment Analysis Pipeline

Builds a sentiment classification pipeline for IMDB reviews using **Logistic Regression**.  
Uses **TF-IDF vectorization**, **text cleaning**, and evaluates performance with accuracy, classification report, and confusion matrix.

---

## 📚 Overview
This version demonstrates an **end-to-end scikit-learn pipeline** that cleans, vectorizes, and classifies movie reviews.  
It’s a modular, reproducible baseline for text classification tasks.

---

## ⚙️ Requirements
pip install pandas scikit-learn

---

## 🧠 Features
- Text cleaning with HTML/punctuation removal  
- TF-IDF vectorization (5k features, unigrams + bigrams)  
- Logistic Regression model  
- Modular design using `Pipeline` and `FunctionTransformer`  
- Emoji sentiment printout for sample reviews  

---

## 🚀 Usage
1. Download IMDB dataset → `data/IMDB Dataset.csv`
2. Run the script:
   python imdb_sentiment_pipeline.py

---

## 📊 Example Output
😊 Positive -> Wonderful movie with great acting.
😠 Negative -> Predictable plot and weak dialogue.

Pipeline Accuracy: 0.8815
              precision    recall  f1-score   support
0             0.87        0.89      0.88      12500
1             0.89        0.87      0.88      12500
[[11125 1375]
 [1550 10950]]

---

## 🧩 Next Steps
- Integrate cross-validation  
- Add hyperparameter optimization  
- Explore different n-gram ranges and vectorizer parameters  
