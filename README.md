\# Sentiment Analysis on Product Reviews



An ML-powered web app that classifies product/movie reviews as

Positive, Negative, or Neutral using TF-IDF + Logistic Regression.



\## Tech Stack

\- Python 3.11

\- scikit-learn (TF-IDF + Logistic Regression)

\- NLTK (Movie Reviews + Twitter Samples dataset)

\- Flask (Web UI)

\- Matplotlib + Seaborn (Confusion Matrix)



\## Dataset

\- NLTK Movie Reviews Corpus — 2000 labelled English reviews

\- NLTK Twitter Samples — 10,000 labelled tweets

\- Total after cleaning — \~10,900 samples



\## Project Structure

sentiment-analysis/

├── app.py               # Flask web application

├── train\_model.py       # ML training script

├── model/               # Saved model files (generated)

├── templates/

│   └── index.html       # Web UI

├── static/              # Confusion matrix image (generated)

├── requirements.txt

└── README.md



\## How to Run

pip install -r requirements.txt

python train\_model.py

python app.py



Then open http://127.0.0.1:5000



\## Model Performance

\- Algorithm  : Logistic Regression

\- Vectorizer : TF-IDF (15,000 features, bigrams)

\- Accuracy   : \~88%

\- F1 Score   : \~88%



\## Limitations

\- Sarcasm detection fails (TF-IDF has no context understanding)

\- English only — Hinglish/Hindi requires MuRIL or XLM-RoBERTa

\- Small dataset compared to production systems



\## Future Improvements

\- Use BERT or MuRIL for multilingual + sarcasm support

\- Add larger dataset (Amazon Reviews, Yelp)

\- Deploy on Heroku or Render

