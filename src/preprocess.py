import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load the SMS Spam Collection dataset.
    The dataset is tab-separated and has no header.
    Columns are 'label' and 'message'.
    """
    try:
        df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_text(text):
    """
    Basic text cleaning:
    - Lowercase
    - Remove punctuation
    - Remove numbers (optional, but good for general text)
    """
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def preprocess_data(df):
    """
    Apply cleaning and return X (features) and y (labels).
    """
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    # encoding labels: spam = 1, ham = 0
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df['cleaned_message'], df['label_num']

if __name__ == "__main__":
    filepath = 'data/SMSSpamCollection'
    df = load_data(filepath)
    if df is not None:
        print("Data loaded successfully.")
        print(df.head())
        
        X, y = preprocess_data(df)
        print("Data processed.")
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        X_tfidf = tfidf.fit_transform(X)
        print(f"TF-IDF Matrix shape: {X_tfidf.shape}")
