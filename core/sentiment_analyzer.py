from textblob import TextBlob
from typing import List, Dict, Any
import re
import pandas as pd
from collections import Counter
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        if pd.isna(text) or not text:
            return ""
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()
        
        return text
    
    def _classify_sentiment(self, polarity: float) -> str:
        """Classify sentiment based on polarity score"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of texts"""
        results = []
        
        for i, text in enumerate(texts):
            cleaned_text = self.preprocess_text(text)
            if cleaned_text:
                blob = TextBlob(cleaned_text)
                
                result = {
                    'index': i,
                    'original_text': text,
                    'sentiment_label': self._classify_sentiment(blob.sentiment.polarity),
                    'polarity': float(blob.sentiment.polarity),
                    'subjectivity': float(blob.sentiment.subjectivity),
                    'confidence': abs(blob.sentiment.polarity)
                }
                results.append(result)
        
        # Create summary
        if results:
            sentiment_counts = Counter([r['sentiment_label'] for r in results])
            avg_polarity = np.mean([r['polarity'] for r in results])
            avg_subjectivity = np.mean([r['subjectivity'] for r in results])
        else:
            sentiment_counts = {}
            avg_polarity = 0.0
            avg_subjectivity = 0.0
        
        return {
            'total_texts': len(results),
            'sentiment_distribution': dict(sentiment_counts),
            'average_polarity': float(avg_polarity),
            'average_subjectivity': float(avg_subjectivity),
            'detailed_results': results
        }
