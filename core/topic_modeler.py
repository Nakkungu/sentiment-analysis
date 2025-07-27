# core/topic_modeler.py

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import pandas as pd
from textblob import TextBlob
import re
import logging

logger = logging.getLogger(__name__)

class TopicModeler:
    """
    Advanced topic modeling class with multiple algorithms support
    """
    
    def __init__(self, 
                 method: str = "lda",
                 n_topics: int = 5,
                 max_features: int = 1000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 random_state: int = 42):
        """
        Initialize the topic modeler
        
        Args:
            method: Topic modeling method ('lda', 'nmf', 'kmeans')
            n_topics: Number of topics to extract
            max_features: Maximum number of features for vectorization
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            random_state: Random state for reproducibility
        """
        self.method = method.lower()
        self.n_topics = n_topics
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for topic modeling
        
        Args:
            texts: List of text documents
            
        Returns:
            List of preprocessed texts
        """
        processed_texts = []
        
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs, mentions, hashtags
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            processed_texts.append(text)
            
        return processed_texts
    
    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Fit the topic model and transform texts
        
        Args:
            texts: List of text documents
            
        Returns:
            Tuple of (document-topic matrix, topics list)
        """
        if not texts:
            raise ValueError("Empty texts list provided")
            
        # Preprocess texts
        processed_texts = self.preprocess_texts(texts)
        
        # Initialize vectorizer based on method
        if self.method in ['lda', 'nmf']:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:  # kmeans
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
            
        # Vectorize texts
        doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Initialize and fit model
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=100
            )
        elif self.method == 'nmf':
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=100
            )
        elif self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_topics,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        # Fit and transform
        if self.method == 'kmeans':
            doc_topic_matrix = self.model.fit_transform(doc_term_matrix.toarray())
        else:
            doc_topic_matrix = self.model.fit_transform(doc_term_matrix)
            
        # Extract topics
        topics = self._extract_topics()
        
        return doc_topic_matrix, topics
    
    def _extract_topics(self, n_words: int = 10) -> List[Dict[str, Any]]:
        """
        Extract topics from the fitted model
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            List of topic dictionaries
        """
        topics = []
        
        if self.method == 'kmeans':
            # For KMeans, use cluster centers
            for topic_idx, cluster_center in enumerate(self.model.cluster_centers_):
                top_words_idx = cluster_center.argsort()[-n_words:][::-1]
                top_words = [self.feature_names[i] for i in top_words_idx]
                weights = [float(cluster_center[i]) for i in top_words_idx]
                
                topics.append({
                    "topic_id": topic_idx,
                    "keywords": top_words,
                    "weights": weights,
                    "coherence_score": float(np.mean(weights))
                })
        else:
            # For LDA and NMF
            for topic_idx, topic in enumerate(self.model.components_):
                top_words_idx = topic.argsort()[-n_words:][::-1]
                top_words = [self.feature_names[i] for i in top_words_idx]
                weights = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    "topic_id": topic_idx,
                    "keywords": top_words,
                    "weights": weights,
                    "coherence_score": float(np.mean(weights))
                })
                
        return topics
    
    def get_document_topics(self, doc_topic_matrix: np.ndarray, 
                           threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get topic assignments for each document
        
        Args:
            doc_topic_matrix: Document-topic probability matrix
            threshold: Minimum probability threshold for topic assignment
            
        Returns:
            List of document topic assignments
        """
        document_topics = []
        
        for doc_idx, topic_probs in enumerate(doc_topic_matrix):
            doc_topics = []
            
            if self.method == 'kmeans':
                # For KMeans, assign to closest cluster
                assigned_topic = int(np.argmin(topic_probs))
                doc_topics.append({
                    "topic_id": assigned_topic,
                    "probability": 1.0  # KMeans gives hard assignments
                })
            else:
                # For LDA and NMF, use probability threshold
                for topic_idx, prob in enumerate(topic_probs):
                    if prob >= threshold:
                        doc_topics.append({
                            "topic_id": topic_idx,
                            "probability": float(prob)
                        })
                        
            document_topics.append({
                "document_id": doc_idx,
                "topics": doc_topics
            })
            
        return document_topics
    
    def analyze_topic_sentiment(self, texts: List[str], 
                               document_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for each topic based on assigned documents
        
        Args:
            texts: Original text documents
            document_topics: Document topic assignments
            
        Returns:
            List of topic sentiment analyses
        """
        topic_sentiments = {}
        
        # Initialize topic sentiment accumulators
        for topic_id in range(self.n_topics):
            topic_sentiments[topic_id] = {
                "sentiments": [],
                "topic_id": topic_id
            }
        
        # Collect sentiments for each topic
        for doc_idx, doc_info in enumerate(document_topics):
            if doc_idx < len(texts):
                # Analyze sentiment of the document
                blob = TextBlob(texts[doc_idx])
                sentiment_score = blob.sentiment.polarity
                
                # Assign sentiment to document's topics
                for topic_info in doc_info["topics"]:
                    topic_id = topic_info["topic_id"]
                    weight = topic_info.get("probability", 1.0)
                    
                    topic_sentiments[topic_id]["sentiments"].append({
                        "score": sentiment_score,
                        "weight": weight
                    })
        
        # Calculate weighted average sentiment for each topic
        topic_sentiment_results = []
        
        for topic_id, data in topic_sentiments.items():
            if data["sentiments"]:
                # Calculate weighted average
                weighted_scores = [s["score"] * s["weight"] for s in data["sentiments"]]
                weights = [s["weight"] for s in data["sentiments"]]
                
                avg_sentiment = sum(weighted_scores) / sum(weights)
                
                # Classify sentiment
                if avg_sentiment > 0.1:
                    sentiment_label = "positive"
                elif avg_sentiment < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                    
                topic_sentiment_results.append({
                    "topic_id": topic_id,
                    "sentiment": sentiment_label,
                    "sentiment_score": float(avg_sentiment),
                    "confidence": float(abs(avg_sentiment)),
                    "document_count": len(data["sentiments"])
                })
            else:
                # No documents assigned to this topic
                topic_sentiment_results.append({
                    "topic_id": topic_id,
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "document_count": 0
                })
                
        return topic_sentiment_results
    
    def get_topic_coherence(self) -> Dict[str, float]:
        """
        Calculate topic coherence metrics (simplified version)
        
        Returns:
            Dictionary with coherence scores
        """
        if self.model is None:
            return {"coherence": 0.0}
            
        # Simplified coherence calculation
        # In a full implementation, you'd use more sophisticated metrics
        coherence_scores = []
        
        if hasattr(self.model, 'components_'):
            for topic in self.model.components_:
                # Calculate variance of topic word weights as a proxy for coherence
                coherence = float(np.var(topic))
                coherence_scores.append(coherence)
        
        return {
            "average_coherence": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            "coherence_scores": coherence_scores
        }