# 💬 Sentiment Analysis Tool

A complete, ready-to-deploy **NLP + Machine Learning** web app that classifies text as
**Positive**, **Negative**, or **Neutral** using TF-IDF features and a Logistic
Regression classifier. Built with **Streamlit**.

> AI/ML Internship Submission Project.

---

## 🚀 Features
- Modern, professional Streamlit UI with gradient header & result cards
- Positive (green) / Negative (red) / Neutral (blue) result cards
- Confidence percentage + per-class probability bars
- Sidebar information panel & professional footer
- Full ML pipeline: text cleaning → stopword removal → TF-IDF → Logistic Regression
- Saves `model.pkl` and `vectorizer.pkl`
- **Auto-trains on first run** if model files are missing (perfect for Streamlit Cloud)

## 📁 Folder Structure
```
Sentiment_Analysis_Tool/
├── app.py              # Streamlit web app
├── train_model.py      # Training pipeline (TF-IDF + Logistic Regression)
├── model.pkl           # Saved trained model (generated)
├── vectorizer.pkl      # Saved TF-IDF vectorizer (generated)
├── requirements.txt
└── README.md
```

## 🛠️ Local Setup
```bash
pip install -r requirements.txt
python train_model.py      # creates model.pkl + vectorizer.pkl
streamlit run app.py
```
Then open the URL shown in your terminal (usually http://localhost:8501).

> You can skip `python train_model.py` — `app.py` will train automatically
> on first launch if the `.pkl` files are missing.

## ☁️ Deploy on Streamlit Cloud
1. Push this folder to a public GitHub repository.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app**, choose your repo, branch, and set the main file to `app.py`.
4. Click **Deploy** — the app will install requirements, auto-train the model,
   and go live as a public web app.

## 🧠 ML Pipeline
1. Load dataset (built-in labeled samples: positive / negative / neutral)
2. Text cleaning (lowercase, strip URLs/punctuation/digits)
3. Stopword removal
4. **TF-IDF Vectorization** (unigrams + bigrams)
5. **Logistic Regression** classifier
6. Evaluate accuracy on a held-out test split
7. Persist `model.pkl` and `vectorizer.pkl`

## 📦 Tech Stack
- Python, Streamlit
- scikit-learn (TF-IDF, Logistic Regression)
- Pandas, NumPy

## 📝 Notes
The dataset is bundled inside `train_model.py` so the project runs **without any
external download or API key**. You can replace `DATA` in `train_model.py` with
a larger CSV (e.g. IMDB movie reviews or Twitter sentiment) for stronger results.

---
Made with ❤️ for the AI/ML Internship submission.
