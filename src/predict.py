import joblib
import sys
from preprocess import clean_text

def predict_spam(message):
    try:
        model = joblib.load('spam_classifier_model.pkl')
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return

    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])[0]
    proba = model.predict_proba([cleaned_message])[0]
    
    label = "Spam" if prediction == 1 else "Ham"
    confidence = proba[prediction]
    
    print(f"Message: {message}")
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    return label, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        predict_spam(message)
    else:
        print("Please provide a message to classify.")
        # Test examples
        print("\n--- Test Examples ---")
        predict_spam("Congratulations! You've won a $1000 gift card. Click here to claim.")
        predict_spam("Hey, are we still on for lunch later?")
