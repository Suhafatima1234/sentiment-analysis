from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle, re, os, io, csv
import pandas as pd
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'sentimentanalysis2024secretkey'
import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database', 'sentiment.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ── Models ───────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    analyses = db.relationship('Analysis', backref='user', lazy=True)

class Analysis(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    text       = db.Column(db.Text, nullable=False)
    sentiment  = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Load ML model ─────────────────────────────────────────────────────────────
model      = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
metadata   = pickle.load(open('model/metadata.pkl', 'rb'))
stop_words = set(stopwords.words('english'))
negation   = {'no','not','nor','never','neither','nobody','nothing','nowhere',
              'without','hardly','barely','scarcely'}
stop_words = stop_words - negation

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def predict(text):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    conf    = round(model.predict_proba(vec).max() * 100, 1)
    return label, conf

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        user   = User(username=username, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user     = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ── Main route ────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET','POST'])
@login_required
def index():
    result = None; review = ''; confidence = None
    if request.method == 'POST':
        review          = request.form['review']
        result, confidence = predict(review)
        entry = Analysis(text=review, sentiment=result,
                         confidence=confidence, user_id=current_user.id)
        db.session.add(entry)
        db.session.commit()
    history = Analysis.query.filter_by(user_id=current_user.id)\
                            .order_by(Analysis.created_at.desc()).limit(5).all()
    return render_template('index.html', result=result, review=review,
                           confidence=confidence, metadata=metadata, history=history)

# ── CSV Upload ────────────────────────────────────────────────────────────────
@app.route('/upload', methods=['GET','POST'])
@login_required
def upload():
    results = None
    if request.method == 'POST':
        file = request.files.get('csvfile')
        if not file or not file.filename.endswith('.csv'):
            flash('Please upload a valid CSV file!', 'danger')
            return redirect(url_for('upload'))
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            flash('CSV must have a column named "text"!', 'danger')
            return redirect(url_for('upload'))
        df = df[df['text'].notna()].head(500)   # max 500 rows
        df['sentiment']  = ''
        df['confidence'] = 0.0
        for i, row in df.iterrows():
            lbl, conf         = predict(str(row['text']))
            df.at[i,'sentiment']  = lbl
            df.at[i,'confidence'] = conf
            entry = Analysis(text=str(row['text'])[:500],
                             sentiment=lbl, confidence=conf,
                             user_id=current_user.id)
            db.session.add(entry)
        db.session.commit()
        results = df[['text','sentiment','confidence']].to_dict('records')
        # save for download
        df.to_csv('static/results.csv', index=False)
    return render_template('upload.html', results=results)

@app.route('/download')
@login_required
def download():
    return send_file('static/results.csv', as_attachment=True,
                     download_name='sentiment_results.csv')

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    all_analyses = Analysis.query.filter_by(user_id=current_user.id).all()
    total     = len(all_analyses)
    positive  = sum(1 for a in all_analyses if a.sentiment == 'positive')
    negative  = sum(1 for a in all_analyses if a.sentiment == 'negative')
    neutral   = sum(1 for a in all_analyses if a.sentiment == 'neutral')
    avg_conf  = round(sum(a.confidence for a in all_analyses) / total, 1) if total else 0
    recent    = Analysis.query.filter_by(user_id=current_user.id)\
                              .order_by(Analysis.created_at.desc()).limit(10).all()
    return render_template('dashboard.html',
                           total=total, positive=positive,
                           negative=negative, neutral=neutral,
                           avg_conf=avg_conf, recent=recent,
                           metadata=metadata)

# ── URL Scraper ───────────────────────────────────────────────────────────────
@app.route('/scrape', methods=['GET','POST'])
@login_required
def scrape():
    results = None; url = ''
    if request.method == 'POST':
        url = request.form['url']
        try:
            import requests
            from bs4 import BeautifulSoup
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp    = requests.get(url, headers=headers, timeout=10)
            soup    = BeautifulSoup(resp.text, 'html.parser')
            # grab all paragraph text as "reviews"
            paras   = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 30][:20]
            if not paras:
                flash('Could not extract text from this URL. Try another.', 'danger')
                return render_template('scrape.html', results=None, url=url)
            results = []
            for para in paras:
                lbl, conf = predict(para)
                results.append({'text': para[:150], 'sentiment': lbl, 'confidence': conf})
                entry = Analysis(text=para[:500], sentiment=lbl,
                                 confidence=conf, user_id=current_user.id)
                db.session.add(entry)
            db.session.commit()
        except Exception as e:
            flash(f'Error scraping URL: {str(e)}', 'danger')
    return render_template('scrape.html', results=results, url=url)

if __name__ == '__main__':
    with app.app_context():
        os.makedirs('database', exist_ok=True)
        db.create_all()
    app.run(debug=True)