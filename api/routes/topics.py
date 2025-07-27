# api/routes/topics.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class TopicRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")
    analyze_topics: bool = Field(default=True, description="Whether to perform topic analysis")
    n_topics: int = Field(default=5, ge=1, le=20, description="Number of topics to extract")
    method: Optional[str] = Field(default="lda", description="Topic modeling method")

class TopicInfo(BaseModel):
    topic_id: int
    keywords: List[str]
    weights: List[float]
    coherence_score: float

class TopicSentiment(BaseModel):
    topic_id: int
    sentiment: str
    sentiment_score: float
    confidence: float
    document_count: int

class DocumentTopic(BaseModel):
    document_id: int
    topics: List[Dict[str, Any]]

class TopicAnalysisResponse(BaseModel):
    topics: List[TopicInfo]
    topic_sentiment_analysis: List[TopicSentiment]
    document_topics: List[DocumentTopic]
    total_topics: int
    method_used: str
    coherence_metrics: Dict[str, float]

@router.post("/analyze-topics", response_model=TopicAnalysisResponse)
async def analyze_topics(request: TopicRequest):
    """
    Analyze topics in the provided texts.
    
    This endpoint performs topic extraction and sentiment analysis.
    """
    
    try:
        # Validate request
        if not request.analyze_topics:
            raise HTTPException(
                status_code=400, 
                detail="analyze_topics must be true for topic analysis endpoint"
            )
            
        # Filter out empty texts
        valid_texts = [text.strip() for text in request.texts if text.strip()]
        
        if len(valid_texts) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid texts provided after filtering empty strings"
            )
            
        # Adjust n_topics if necessary
        if len(valid_texts) < request.n_topics:
            n_topics = len(valid_texts)
        else:
            n_topics = request.n_topics
        
        logger.info(f"Processing {len(valid_texts)} texts for {n_topics} topics")
        
        # Simulate topic analysis - replace this with real implementation later
        topics = []
        topic_sentiments = []
        document_topics = []
        
        # Generate mock topics based on common themes
        topic_keywords = [
            ["healthcare", "medical", "insurance", "costs", "care"],
            ["energy", "renewable", "climate", "solar", "environment"], 
            ["education", "students", "schools", "teachers", "learning"],
            ["economy", "jobs", "employment", "wages", "economic"],
            ["government", "policy", "political", "legislation", "public"]
        ]
        
        topic_names = ["Healthcare", "Energy/Environment", "Education", "Economy", "Government"]
        
        for i in range(n_topics):
            # Create topic info
            keywords = topic_keywords[i % len(topic_keywords)]
            weights = [0.9 - (j * 0.1) for j in range(len(keywords))]
            
            topics.append(TopicInfo(
                topic_id=i,
                keywords=keywords,
                weights=weights,
                coherence_score=0.7 + (i * 0.05)
            ))
            
            # Create topic sentiment (simulate based on keywords)
            if "healthcare" in keywords or "costs" in keywords:
                sentiment = "negative"
                sentiment_score = -0.3
            elif "renewable" in keywords or "solar" in keywords:
                sentiment = "positive" 
                sentiment_score = 0.4
            else:
                sentiment = "neutral"
                sentiment_score = 0.1
                
            topic_sentiments.append(TopicSentiment(
                topic_id=i,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                confidence=0.8,
                document_count=len(valid_texts) // n_topics
            ))
        
        # Assign documents to topics
        for doc_idx in range(len(valid_texts)):
            assigned_topic = doc_idx % n_topics
            document_topics.append(DocumentTopic(
                document_id=doc_idx,
                topics=[{"topic_id": assigned_topic, "probability": 0.8}]
            ))
        
        # Calculate coherence metrics
        coherence_metrics = {
            "average_coherence": 0.75,
            "topic_coherence": [0.7 + (i * 0.05) for i in range(n_topics)]
        }
        
        logger.info(f"Topic analysis completed successfully. Generated {len(topics)} topics")
        
        return TopicAnalysisResponse(
            topics=topics,
            topic_sentiment_analysis=topic_sentiments,
            document_topics=document_topics,
            total_topics=n_topics,
            method_used=request.method or "lda",
            coherence_metrics=coherence_metrics
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in topic analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Topic analysis failed: {str(e)}")

@router.get("/topics/methods")
async def get_available_methods():
    """Get available topic modeling methods"""
    return {
        "methods": {
            "lda": {
                "name": "Latent Dirichlet Allocation",
                "description": "Probabilistic topic modeling",
                "best_for": "General topic discovery"
            },
            "nmf": {
                "name": "Non-negative Matrix Factorization", 
                "description": "Matrix factorization for topics",
                "best_for": "Clear topic separation"
            },
            "kmeans": {
                "name": "K-Means Clustering",
                "description": "Clustering-based topics",
                "best_for": "Hard topic assignments"
            }
        },
        "default_method": "lda"
    }