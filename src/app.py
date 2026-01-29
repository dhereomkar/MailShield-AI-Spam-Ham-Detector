from flask import Flask, render_template, request, jsonify
import joblib
from preprocess import clean_text

app = Flask(__name__)

# Load model (lazy loading or at startup)
try:
    model = joblib.load('spam_classifier_model.pkl')
except FileNotFoundError:
    model = None
    print("Error: content model not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
        
    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])[0]
    proba = model.predict_proba([cleaned_message])[0]
    
    label = "Spam" if prediction == 1 else "Ham"
    confidence = float(proba[prediction])
    
    return jsonify({
        'prediction': label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
