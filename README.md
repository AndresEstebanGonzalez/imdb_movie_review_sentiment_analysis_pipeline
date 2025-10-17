🎬 IMDB Sentiment Analysis Pipeline

Builds a sentiment classification pipeline for IMDB movie reviews using Logistic Regression.
Performs text cleaning, TF-IDF vectorization, model training, and evaluation, with both console and saved report outputs.
The trained pipeline is automatically stored for reuse or deployment.

⸻

📚 Overview

This version demonstrates a complete end-to-end scikit-learn pipeline that:
	•	Cleans raw review text
	•	Converts text into numerical TF-IDF features
	•	Trains a Logistic Regression classifier
	•	Evaluates model performance
	•	Saves both the trained pipeline and an evaluation report

It serves as a reproducible and extensible foundation for text classification or sentiment analysis projects.

⸻

⚙️ Requirements

pip install pandas numpy scikit-learn joblib


⸻

🧠 Features
	•	✅ Text cleaning (HTML & punctuation removal)
	•	✅ TF-IDF vectorization (5k features, unigrams + bigrams)
	•	✅ Logistic Regression model (Liblinear solver)
	•	✅ Modular design with Pipeline + FunctionTransformer
	•	✅ Sample sentiment printout with emojis
	•	✅ Automatic model saving (.joblib)
	•	✅ Auto-generated evaluation report (.txt)

⸻

🗂️ Project Structure

.
├── data/
│   └── IMDB Dataset.csv
├── reports/
│   └── pipeline_evaluation.txt
├── imdb_sentiment_pipeline.py
└── imdb_logreg_pipeline.joblib


⸻

🚀 Usage
	1.	Download the IMDB dataset → data/IMDB Dataset.csv
	2.	Run the script:

python imdb_sentiment_pipeline.py


	3.	After execution:
	•	The trained pipeline will be saved to
imdb_logreg_pipeline.joblib
	•	Evaluation results will be saved to
reports/pipeline_evaluation.txt

⸻

📊 Example Output

😊 Positive -> Wonderful movie with great acting.
😠 Negative -> Predictable plot and weak dialogue.

=== Logistic Regression ===
Accuracy: 0.8815

Classification Report:
              precision    recall  f1-score   support
0             0.87        0.89      0.88      12500
1             0.89        0.87      0.88      12500

Confusion Matrix:
[[11125 1375]
 [1550 10950]]


⸻

🧩 Next Steps
	•	Add cross-validation
	•	Implement hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
	•	Compare with other classifiers (Naive Bayes, SVM, etc.)
	•	Build an inference script or simple API for predictions
