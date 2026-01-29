import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_data, preprocess_data

def train_models():
    # Load and preprocess data
    filepath = 'data/SMSSpamCollection'
    df = load_data(filepath)
    if df is None:
        return

    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Naive Bayes
    print("Training Naive Bayes...")
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
        ('nb', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)
    y_pred_nb = nb_pipeline.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb))
    
    # Model 2: SVM
    print("Training SVM...")
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
        ('svm', SVC(kernel='linear', probability=True))
    ])
    svm_pipeline.fit(X_train, y_train)
    y_pred_svm = svm_pipeline.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))
    
    # Save the best model (usually SVM performs slightly better or similar)
    if accuracy_score(y_test, y_pred_svm) >= accuracy_score(y_test, y_pred_nb):
        best_model = svm_pipeline
        print("Saving SVM model...")
        joblib.dump(best_model, 'spam_classifier_model.pkl')
    else:
        best_model = nb_pipeline
        print("Saving Naive Bayes model...")
        joblib.dump(best_model, 'spam_classifier_model.pkl')
        
    print("Model saved as spam_classifier_model.pkl")

if __name__ == "__main__":
    train_models()
