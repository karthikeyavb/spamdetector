import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # <-- Changed algorithm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK data (run this once)
nltk.download('stopwords')

def text_process(mess):
    """
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Custom tokenizer wrapper for Tfidf
def dummy_fun(doc):
    return doc

def train_model_advanced():
    print("Loading and processing data...")
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df.drop_duplicates(inplace=True)

    # APPLY PREPROCESSING
    # This might take a moment
    print("Cleaning text (removing stopwords)...")
    df['clean_msg'] = df['message'].apply(text_process)

    X = df['clean_msg']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Advanced SVM Model...")
    # We use SVC (Support Vector Classifier) instead of Naive Bayes
    # It is generally much better at drawing a distinct line between spam and ham
    model = make_pipeline(
        TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None), 
        SVC(kernel='sigmoid', gamma=1.0)
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions))

    joblib.dump(model, 'models/spam_model.pkl')
    print("Advanced model saved!")

if __name__ == "__main__":
    train_model_advanced()