from flask import Flask, render_template, request
import pickle
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)

model      = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
metadata   = pickle.load(open('model/metadata.pkl', 'rb'))
stop_words = set(stopwords.words('english'))

history = []   # in-session history

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    result     = None
    review     = ''
    confidence = None

    if request.method == 'POST':
        review  = request.form['review']
        cleaned = clean_text(review)
        vec     = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba      = model.predict_proba(vec).max() * 100
        result     = prediction
        confidence = round(proba, 1)
        history.insert(0, {
            'text'      : review[:80] + ('...' if len(review) > 80 else ''),
            'label'     : result,
            'confidence': confidence
        })

    return render_template('index.html',
                           result=result,
                           review=review,
                           confidence=confidence,
                           metadata=metadata,
                           history=history[:5])

if __name__ == '__main__':
    app.run(debug=True)