from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer 
import re

app = Flask(__name__)

# Load the summarization model
print("Loading the summarization model...")
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print("Error loading model:", e)
    exit(1)

# Function to extract keywords using TF-IDF
def extract_keywords(text, num_keywords=3):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the top keywords and their scores
    keywords = []
    scores = tfidf_matrix.toarray()[0]
    keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    
    return [keyword for keyword, score in keyword_scores[:num_keywords]]

# Function to find context for a keyword in the original text
def get_keyword_context(input_text, keyword):
    sentences = re.split(r'(?<=[.!?])\s+', input_text)
    for sentence in sentences:
        if keyword.lower() in sentence.lower():
            return sentence.strip()
    return "Context not found."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Get the input text from the request
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate the summary
        summary = summarizer(input_text, max_length=75, min_length=25, do_sample=False)
        summary_text = summary[0]['summary_text']

        # Extract keywords from the summary
        keywords = extract_keywords(summary_text, num_keywords=3)

        # Get context for each keyword from the input text
        keyword_contexts = {keyword: get_keyword_context(input_text, keyword) for keyword in keywords}

        return jsonify({
            'summary': summary_text,
            'keywords': keywords,
            'keyword_contexts': keyword_contexts
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)