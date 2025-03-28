import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("data-science-tweets.csv", encoding="utf-8")

# Ensure required columns exist
if "text" not in df.columns or "airline_sentiment" not in df.columns:
    raise ValueError("Required columns 'text' and 'airline_sentiment' not found in dataset")

# Text Cleaning Function
def clean_tweet(tweet):
    tweet = str(tweet).lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    words = word_tokenize(tweet)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply text cleaning
df["cleaned_tweet"] = df["text"].apply(clean_tweet)

# Convert text to numerical (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_tweet"]).toarray()
y = df["airline_sentiment"].map({"positive": 1, "negative": 0, "neutral": 2})

# Drop NaN values if any
df.dropna(subset=["cleaned_tweet", "airline_sentiment"], inplace=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save Model & Vectorizer
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\u2705 Model and Vectorizer saved successfully!")
