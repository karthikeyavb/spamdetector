import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from utils import transform_text # Importing our custom cleaner

def train_advanced():
    print("1. Loading Data...")
    try:
        df = pd.read_csv('data/spam.csv', encoding='latin-1')
    except FileNotFoundError:
        print("Error: data/spam.csv not found.")
        return

    # Basic Cleanup
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
    df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
    
    # Encode Labels (Spam = 1, Ham = 0)
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])
    
    # Remove Duplicates
    df = df.drop_duplicates(keep='first')

    print("2. Preprocessing Text (This may take a moment)...")
    # We apply the cleaning from utils.py
    df['transformed_text'] = df['text'].apply(transform_text)

    X = df['transformed_text']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- THE ADVANCED PART ---
    # We build a pipeline with SVM
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)), # Limit to top 3000 words to reduce noise
        ('svm', SVC(kernel='sigmoid', probability=True)) # Sigmoid kernel works best for binary text classification
    ])

    # GridSearch: It will try different 'C' (Strictness) and 'gamma' values
    # to find the absolute best math for this specific dataset.
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': [0.1, 1, 'scale']
    }

    print("3. Training & Tuning Hyperparameters (GridSearch)...")
    print("   (This is training multiple models to find the best one. Please wait.)")
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"   Best Parameters found: {grid.best_params_}")
    best_model = grid.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)
    print("\n--- Model Accuracy ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Save the best model
    print("4. Saving the best model...")
    joblib.dump(best_model, 'models/spam_model_advanced.pkl')
    print("Done! Model saved to models/spam_model_advanced.pkl")

if __name__ == "__main__":
    train_advanced()