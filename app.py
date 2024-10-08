from flask import Flask, request, jsonify, render_template
import joblib
import re

# Initialize Flask app
app = Flask(__name__)

# Load the saved model, TF-IDF vectorizer, and label encoder
model = joblib.load('model/svm_model.joblib')
tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
label_encoder = joblib.load('model/label_encoder.joblib')

# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()             # Convert to lowercase
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    text = text.strip()             # Strip leading and trailing spaces
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the movie summary from the request
    summary = request.form.get('summary', '')

    # Clean and vectorize the input summary
    cleaned_summary = clean_text(summary)
    vectorized_summary = tfidf_vectorizer.transform([cleaned_summary])

    # Predict the genre
    prediction = model.predict(vectorized_summary)
    genre = label_encoder.inverse_transform(prediction)

    return jsonify({'genre': genre[0]})

if __name__ == '__main__':
    app.run(debug=True)
