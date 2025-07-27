from fastapi import APIRouter, HTTPException
from app.models import TextAnalysisRequest, SentimentResponse
from core.sentiment_analyzer import SentimentAnalyzer

router = APIRouter()
analyzer = SentimentAnalyzer()

@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextAnalysisRequest):
    """Analyze sentiment of political texts"""
    try:
        results = analyzer.analyze_sentiment(request.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in sentiment analysis: {str(e)}")
