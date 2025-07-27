from fastapi import APIRouter, HTTPException
from app.models import TextAnalysisRequest, SentimentResponse
from core.political_analyzer import PoliticalSentimentAnalyzer  # Use your analyzer

router = APIRouter()
analyzer = PoliticalSentimentAnalyzer()  # Your comprehensive analyzer

@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextAnalysisRequest):
    """Analyze sentiment of political texts"""
    try:
        # Convert to use your analyzer's method
        sentiment_df = analyzer.analyze_sentiment(request.texts)
        
        # Convert DataFrame to the expected response format
        results = []
        for idx, row in sentiment_df.iterrows():
            results.append({
                'index': idx,
                'original_text': row['text'],
                'sentiment_label': row['sentiment_label'],
                'polarity': row['polarity'],
                'subjectivity': row['subjectivity'],
                'confidence': abs(row['polarity'])
            })
        
        # Calculate summary statistics
        sentiment_counts = sentiment_df['sentiment_label'].value_counts().to_dict()
        
        return {
            'total_texts': len(results),
            'sentiment_distribution': sentiment_counts,
            'average_polarity': float(sentiment_df['polarity'].mean()),
            'average_subjectivity': float(sentiment_df['subjectivity'].mean()),
            'detailed_results': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in sentiment analysis: {str(e)}")