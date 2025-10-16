"""
IMDB Sentiment Analysis Pipeline

Builds a sentiment classification pipeline for IMDB reviews using
Logistic Regression. Includes text cleaning, TF-IDF vectorization,
and model evaluation through accuracy, classification report,
and confusion matrix.
"""

#Import re
import re
#Import Pandas
import pandas as pd
#Import split
from sklearn.model_selection import train_test_split
#Import Pipeline
from sklearn.pipeline import Pipeline
#Import model
from sklearn.linear_model import LogisticRegression
#Import preprocessing
from sklearn.preprocessing import FunctionTransformer
#Import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Import data
DATA_PATH = "data/IMDB Dataset.csv"
imdb_df_full = pd.read_csv(DATA_PATH, encoding="latin-1", header=None)
imdb_df = imdb_df_full.drop(index=0)
imdb_df.columns = ["review", "polarity"]
#Convert positive & negative to 1 & 0
imdb_df["polarity"] = imdb_df["polarity"].map({"positive":1, "negative":0})
#Split data
X_train, X_test, y_train, y_test = train_test_split(
    imdb_df["review"],
    imdb_df["polarity"],
    stratify=imdb_df["polarity"]
)
#Clean text batch function
def clean_text_batch(reviews):
    '''Clean a list (batch) of text reviews.'''
    cleaned = []
    for text in reviews:
        text = text.strip()
        text = re.sub(r"<.*?>", "",text)
        text = re.sub(r"[^a-z0-9'\s]", "", text)
        cleaned.append(text)
    return cleaned
#Pipeline
pipeline = Pipeline(steps=[
    ("cleaner", FunctionTransformer(clean_text_batch, validate=False)),
    ("vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1,2), lowercase=False)),
    ("model", LogisticRegression(max_iter=100, random_state=1,solver="liblinear"))
])
#Fit and predict
pipeline.fit(X_train, y_train)
pipeline_prediction = pipeline.predict(X_test)

#Print review emoji
SAMPLE_PRINT = True
N = 3
if SAMPLE_PRINT:
    sample = imdb_df.sample(N, random_state=1)
    for review, polarity in zip(sample["review"], sample["polarity"]):
        SENTIMENT = "😊 Positive" if polarity == 1 else "😠 Negative"
        print(f"{SENTIMENT} -> {review}")

print("Pipeline Accuracy:", accuracy_score(y_test, pipeline_prediction))
print(classification_report(y_test, pipeline_prediction))
print(confusion_matrix(y_test, pipeline_prediction))
