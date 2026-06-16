"""
train_model.py
Trains a sentiment classifier (Positive / Negative / Neutral) using
TF-IDF features + Logistic Regression, and saves model.pkl + vectorizer.pkl.

Run:
    python train_model.py
"""

import os
import re
import pickle
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# 1. Built-in dataset (so the project runs without any external download).
#    Labels: positive, negative, neutral
# ---------------------------------------------------------------------------
DATA = [
    # Positive
    ("I love this movie, it was absolutely fantastic and heart-warming.", "positive"),
    ("What a wonderful performance, truly brilliant acting!", "positive"),
    ("Amazing storyline and beautiful cinematography. Loved it.", "positive"),
    ("This product works perfectly, I am very happy with it.", "positive"),
    ("Best purchase I have made this year, highly recommended!", "positive"),
    ("The food was delicious and the staff were super friendly.", "positive"),
    ("An incredible experience, I enjoyed every single moment.", "positive"),
    ("Great quality, fast delivery, and excellent customer service.", "positive"),
    ("Absolutely loved the design and the smooth user experience.", "positive"),
    ("This is the best app I have ever used, super intuitive!", "positive"),
    ("Fantastic results, exceeded all my expectations.", "positive"),
    ("Wonderful trip with breathtaking views and friendly people.", "positive"),
    ("She gave an outstanding speech that inspired everyone.", "positive"),
    ("The new update is awesome, everything feels faster.", "positive"),
    ("Brilliant book, I could not put it down.", "positive"),
    ("Happy customer here, will definitely buy again.", "positive"),
    ("Superb sound quality and very comfortable to wear.", "positive"),
    ("The team did an excellent job on this project.", "positive"),
    ("I am thrilled with the outcome, simply perfect.", "positive"),
    ("Lovely atmosphere and amazing music throughout the night.", "positive"),

    # Negative
    ("I hate this movie, it was a complete waste of time.", "negative"),
    ("Terrible acting and a boring, predictable storyline.", "negative"),
    ("Worst purchase ever, broke within two days.", "negative"),
    ("The food was cold, tasteless, and overpriced.", "negative"),
    ("Awful customer service, nobody responded to my emails.", "negative"),
    ("This app keeps crashing, it is incredibly frustrating.", "negative"),
    ("Very disappointed with the quality, not worth the money.", "negative"),
    ("Horrible experience, I will never come back here again.", "negative"),
    ("The product arrived damaged and support refused to help.", "negative"),
    ("Poor design and confusing interface, hated using it.", "negative"),
    ("The new update is terrible, it ruined the whole app.", "negative"),
    ("Boring book, I could barely finish the first chapter.", "negative"),
    ("Bad sound quality and uncomfortable after ten minutes.", "negative"),
    ("Slow, buggy, and full of annoying advertisements.", "negative"),
    ("Rude staff and dirty rooms, would not recommend.", "negative"),
    ("Completely useless, does not do what it promises.", "negative"),
    ("I regret buying this, total disappointment.", "negative"),
    ("The plot was dull and the ending made no sense.", "negative"),
    ("Cheap materials and falls apart easily, very bad.", "negative"),
    ("I am angry and frustrated with how this was handled.", "negative"),

    # Neutral
    ("The movie was okay, nothing special but not bad either.", "neutral"),
    ("It is an average product, does the job as expected.", "neutral"),
    ("The package arrived on the scheduled delivery date.", "neutral"),
    ("The book has 300 pages and is divided into 12 chapters.", "neutral"),
    ("The meeting is scheduled for tomorrow at 10 am.", "neutral"),
    ("This phone has a 6 inch screen and 128 GB of storage.", "neutral"),
    ("The store opens at 9 and closes at 9 every day.", "neutral"),
    ("He walked to the office and started his computer.", "neutral"),
    ("The report contains data from the last three quarters.", "neutral"),
    ("It is a standard laptop with regular specifications.", "neutral"),
    ("The train departs from platform number four.", "neutral"),
    ("The class will cover chapters five through seven.", "neutral"),
    ("She works as a software engineer at a tech company.", "neutral"),
    ("The film is two hours long and rated PG-13.", "neutral"),
    ("The form requires your name, email, and phone number.", "neutral"),
    ("The temperature today is around twenty degrees.", "neutral"),
    ("The hotel has fifty rooms across three floors.", "neutral"),
    ("This document explains the basic setup process.", "neutral"),
    ("The bus route passes through the city center.", "neutral"),
    ("He answered the question and sat back down.", "neutral"),
]

# Minimal built-in stopword list (avoids NLTK download dependency).
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
    "of", "at", "by", "for", "to", "from", "in", "on", "off", "over", "under",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "have", "has", "had", "having", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "this", "that", "these", "those", "as", "so",
    "than", "too", "very", "can", "will", "just", "should", "now", "also",
    "into", "about", "there", "here", "then",
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits, remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def main() -> None:
    df = pd.DataFrame(DATA, columns=["text", "label"])
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=4.0)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds))

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(here, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("Saved model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    main()
