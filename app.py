from flask import Flask, render_template, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Initialize NLTK's Vader Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to analyze sentiment
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()  # Get data from frontend
    text = data.get('text', '')  # Extract text input

    # Get the sentiment score using VADER
    sentiment_score = sia.polarity_scores(text)
    sentiment = "Neutral"
    if sentiment_score['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_score['compound'] <= -0.05:
        sentiment = "Negative"
    
    # Return the sentiment and score
    return jsonify({
        'sentiment': sentiment,
        'score': sentiment_score['compound']
    })

if __name__ == '__main__':
    app.run(debug=True)
