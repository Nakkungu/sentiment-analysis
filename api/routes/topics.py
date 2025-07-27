# Replace the content in api/routes/topics.py with this:

from fastapi import APIRouter, HTTPException
from app.models import TextAnalysisRequest, TopicResponse
from core.political_analyzer import PoliticalSentimentAnalyzer

router = APIRouter()

@router.post("/analyze-topics", response_model=TopicResponse)
async def analyze_topics(request: TextAnalysisRequest):
    """Perform topic modeling and sentiment analysis on political texts"""
    try:
        # Initialize analyzer
        analyzer = PoliticalSentimentAnalyzer()
        
        # Validate input
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 texts are required for topic modeling"
            )
        
        # Set number of topics (with reasonable limits)
        n_topics = min(max(request.n_topics or 5, 2), len(request.texts))
        
        # Perform topic modeling
        topics, topic_matrix = analyzer.perform_topic_modeling(
            request.texts, 
            n_topics=n_topics
        )
        
        if topics is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to perform topic modeling. Texts may be too short or similar."
            )
        
        # Perform sentiment analysis for topic-sentiment correlation
        sentiment_results = analyzer.analyze_sentiment(request.texts)
        
        # Analyze sentiment by topic
        topic_sentiment_analysis = analyzer.analyze_sentiment_by_topic(
            sentiment_results, topics, topic_matrix
        )
        
        # Format response
        response = {
            "topics": topics,
            "topic_sentiment_analysis": topic_sentiment_analysis,
            "total_topics": len(topics)
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error in topic analysis: {str(e)}"
        )

@router.post("/analyze-combined")
async def analyze_combined(request: TextAnalysisRequest):
    """Perform combined sentiment and topic analysis"""
    try:
        analyzer = PoliticalSentimentAnalyzer()
        
        # Perform sentiment analysis
        sentiment_results = analyzer.analyze_sentiment(request.texts)
        
        # Format sentiment results
        sentiment_summary = {
            "total_texts": len(sentiment_results),
            "sentiment_distribution": sentiment_results['sentiment_label'].value_counts().to_dict(),
            "average_polarity": float(sentiment_results['polarity'].mean()),
            "average_subjectivity": float(sentiment_results['subjectivity'].mean()),
            "detailed_results": sentiment_results.to_dict('records')
        }
        
        # Perform topic analysis if requested
        topic_results = None
        if request.analyze_topics and len(request.texts) >= 2:
            n_topics = min(max(request.n_topics or 5, 2), len(request.texts))
            topics, topic_matrix = analyzer.perform_topic_modeling(
                request.texts, n_topics=n_topics
            )
            
            if topics:
                topic_sentiment_analysis = analyzer.analyze_sentiment_by_topic(
                    sentiment_results, topics, topic_matrix
                )
                
                topic_results = {
                    "topics": topics,
                    "topic_sentiment_analysis": topic_sentiment_analysis,
                    "total_topics": len(topics)
                }
        
        return {
            "sentiment_analysis": sentiment_summary,
            "topic_analysis": topic_results,
            "combined_insights": {
                "most_positive_topic": _get_most_positive_topic(topic_results),
                "most_negative_topic": _get_most_negative_topic(topic_results),
                "topic_diversity": len(topic_results["topics"]) if topic_results else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in combined analysis: {str(e)}"
        )

def _get_most_positive_topic(topic_results):
    """Helper function to find most positive topic"""
    if not topic_results or not topic_results.get("topic_sentiment_analysis"):
        return None
    
    max_polarity = -2
    most_positive = None
    
    for topic, analysis in topic_results["topic_sentiment_analysis"].items():
        if analysis["average_polarity"] > max_polarity:
            max_polarity = analysis["average_polarity"]
            most_positive = topic
    
    return most_positive

def _get_most_negative_topic(topic_results):
    """Helper function to find most negative topic"""
    if not topic_results or not topic_results.get("topic_sentiment_analysis"):
        return None
    
    min_polarity = 2
    most_negative = None
    
    for topic, analysis in topic_results["topic_sentiment_analysis"].items():
        if analysis["average_polarity"] < min_polarity:
            min_polarity = analysis["average_polarity"]
            most_negative = topic
    
    return most_negative