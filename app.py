from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Load saved model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    words = word_tokenize(tweet)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Homepage Route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    tweet = request.form["tweet"]
    cleaned_tweet = clean_tweet(tweet)
    vectorized_tweet = vectorizer.transform([cleaned_tweet]).toarray()
    prediction = model.predict(vectorized_tweet)
    
    sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
    return render_template("index.html", tweet=tweet, sentiment=sentiment)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
