from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Request Models
class TextAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    analyze_topics: Optional[bool] = Field(default=False, description="Whether to perform topic modeling")
    n_topics: Optional[int] = Field(default=5, description="Number of topics for topic modeling")

class InteractionData(BaseModel):
    source_user: str
    target_user: str
    interaction_type: str = Field(default="mention", description="Type of interaction")
    weight: float = Field(default=1.0, description="Weight/strength of the interaction")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of interaction")

class NetworkAnalysisRequest(BaseModel):
    interactions: List[InteractionData]
    calculate_influence: Optional[bool] = Field(default=True, description="Whether to calculate influence metrics")
    top_n: Optional[int] = Field(default=20, description="Number of top influencers to return")

# Response Models
class SentimentResult(BaseModel):
    index: int
    original_text: str
    sentiment_label: str
    polarity: float
    subjectivity: float
    confidence: float

class SentimentResponse(BaseModel):
    total_texts: int
    sentiment_distribution: Dict[str, int]
    average_polarity: float
    average_subjectivity: float
    detailed_results: List[SentimentResult]

class InfluencerData(BaseModel):
    user: str
    influence_score: float
    pagerank: float
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    in_degree: int
    out_degree: int

class InfluenceResponse(BaseModel):
    total_users: int
    total_connections: int
    network_density: float
    top_influencers: List[InfluencerData]
    network_stats: Dict[str, Any]

class TopicResponse(BaseModel):
    topics: Dict[str, List[str]]
    topic_sentiment_analysis: Dict[str, Dict[str, Any]]
    total_topics: int

class VisualizationResponse(BaseModel):
    image: str  # base64 encoded image
    format: str
    encoding: str
    metadata: Dict[str, Any]
