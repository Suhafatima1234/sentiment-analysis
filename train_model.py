import nltk
import re
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

nltk.download('movie_reviews')
nltk.download('twitter_samples')
nltk.download('stopwords')

from nltk.corpus import movie_reviews, twitter_samples, stopwords

print("✅ Datasets loaded")

# ── Build dataframe ──────────────────────────────────────────────────────────
records = []

# Positive — movie reviews (1000)
for fileid in movie_reviews.fileids('pos'):
    records.append({'text': movie_reviews.raw(fileid), 'label': 'positive'})

# Negative — movie reviews (1000)
for fileid in movie_reviews.fileids('neg'):
    records.append({'text': movie_reviews.raw(fileid), 'label': 'negative'})

# Positive tweets (strong signal)
pos_tweets = twitter_samples.strings('positive_tweets.json')
for t in pos_tweets:
    records.append({'text': t, 'label': 'positive'})

# Negative tweets (strong signal)
neg_tweets = twitter_samples.strings('negative_tweets.json')
for t in neg_tweets:
    records.append({'text': t, 'label': 'negative'})

# Neutral — ONLY use clearly neutral/objective sentences
# We build neutral from the objective twitter file but filter carefully
neutral_raw = twitter_samples.strings('tweets.20150430-223406.json')
neutral_clean_candidates = []
for t in neutral_raw:
    t_lower = t.lower()
    # Skip if it has strong positive/negative signal words
    positive_words = ['love','great','amazing','awesome','excellent','fantastic',
                      'wonderful','best','happy','good','nice','super','perfect']
    negative_words = ['hate','terrible','awful','horrible','worst','bad','disgusting',
                      'useless','broken','disappointing','poor','sucks','trash']
    has_pos = any(w in t_lower for w in positive_words)
    has_neg = any(w in t_lower for w in negative_words)
    if not has_pos and not has_neg and len(t) > 30:
        neutral_clean_candidates.append(t)

# Take only 800 neutral to keep balance (positive/negative are ~2000 each)
for t in neutral_clean_candidates[:800]:
    records.append({'text': t, 'label': 'neutral'})

df = pd.DataFrame(records)
print(f"\n📊 Dataset breakdown:")
print(df['label'].value_counts().to_string())
print(f"   Total: {len(df)} samples")

# ── Preprocessing ────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
# Keep negation words — critical for sentiment!
negation_words = {'no','not','nor','never','neither','nobody','nothing',
                  'nowhere','neither','without','hardly','barely','scarcely'}
stop_words = stop_words - negation_words

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df['clean'] = df['text'].apply(clean_text)
df = df[df['clean'].str.len() > 10].reset_index(drop=True)
print(f"✅ After cleaning: {len(df)} samples")

# ── Vectorize ────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

X = vectorizer.fit_transform(df['clean'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📦 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Train ────────────────────────────────────────────────────────────────────
print("\n🏋️  Training...")
model = LogisticRegression(
    max_iter=1000,
    C=5.0,
    solver='lbfgs'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')

print(f"\n✅ Accuracy : {acc:.4f}")
print(f"✅ F1 Score : {f1:.4f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ── Confusion matrix ─────────────────────────────────────────────────────────
os.makedirs('static', exist_ok=True)
cm = confusion_matrix(y_test, y_pred, labels=['positive','negative','neutral'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['positive','negative','neutral'],
            yticklabels=['positive','negative','neutral'])
plt.title('Confusion Matrix — Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=120)
plt.close()
print("📊 Confusion matrix saved")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
pickle.dump(model,      open('model/model.pkl',      'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

metadata = {
    'model_name'    : 'Logistic Regression',
    'accuracy'      : round(acc * 100, 2),
    'f1_score'      : round(f1  * 100, 2),
    'train_size'    : X_train.shape[0],
    'test_size'     : X_test.shape[0],
    'total_samples' : len(df),
    'dataset'       : 'NLTK Movie Reviews + Twitter Samples',
    'features'      : 15000,
    'ngrams'        : '(1,2)'
}
pickle.dump(metadata, open('model/metadata.pkl', 'wb'))

print(f"\n{'='*50}")
print(f"  Model    : {metadata['model_name']}")
print(f"  Data     : {metadata['total_samples']} samples")
print(f"  Accuracy : {metadata['accuracy']}%")
print(f"  F1 Score : {metadata['f1_score']}%")
print(f"{'='*50}")
print("\n✅ All saved! Run: python app.py")