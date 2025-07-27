# Add this to your api/routes/upload.py file

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
from core.political_analyzer import PoliticalSentimentAnalyzer
from typing import Optional
import json

router = APIRouter()

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for analysis"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        return {
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "sample_data": df.head().to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@router.post("/analyze-csv-sentiment")
async def analyze_csv_sentiment(
    file: UploadFile = File(...),
    text_column: str = Form(..., description="Name of the column containing text to analyze"),
    id_column: Optional[str] = Form(None, description="Optional ID column name"),
    analyze_topics: bool = Form(False, description="Whether to perform topic modeling"),
    n_topics: int = Form(5, description="Number of topics for topic modeling")
):
    """
    Upload CSV and analyze sentiment of specified text column
    
    Parameters:
    - file: CSV file to upload
    - text_column: Name of column containing text to analyze
    - id_column: Optional column to use as identifier
    - analyze_topics: Whether to perform topic modeling
    - n_topics: Number of topics to extract
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
            )
        
        # Initialize analyzer
        analyzer = PoliticalSentimentAnalyzer()
        
        # Extract texts
        texts = df[text_column].dropna().astype(str).tolist()
        
        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="No valid text data found")
        
        # Perform sentiment analysis
        sentiment_results = analyzer.analyze_sentiment(texts)
        
        # Add original row identifiers if specified
        if id_column and id_column in df.columns:
            # Map back to original data
            valid_indices = df[text_column].dropna().index
            ids = df.loc[valid_indices, id_column].tolist()
            sentiment_results['original_id'] = ids[:len(sentiment_results)]
        
        # Topic modeling if requested
        topics_result = None
        topic_sentiment_analysis = None
        
        if analyze_topics and len(texts) >= n_topics:
            topics, topic_matrix = analyzer.perform_topic_modeling(texts, n_topics)
            if topics:
                topics_result = topics
                topic_sentiment_analysis = analyzer.analyze_sentiment_by_topic(
                    sentiment_results, topics, topic_matrix
                )
        
        # Prepare response
        response = {
            "file_info": {
                "filename": file.filename,
                "total_rows": len(df),
                "analyzed_rows": len(texts),
                "text_column": text_column
            },
            "sentiment_analysis": {
                "total_texts": len(sentiment_results),
                "sentiment_distribution": sentiment_results['sentiment_label'].value_counts().to_dict(),
                "average_polarity": float(sentiment_results['polarity'].mean()),
                "average_subjectivity": float(sentiment_results['subjectivity'].mean()),
                "detailed_results": sentiment_results.to_dict('records')
            }
        }
        
        if topics_result:
            response["topic_analysis"] = {
                "topics": topics_result,
                "topic_sentiment_analysis": topic_sentiment_analysis,
                "total_topics": len(topics_result)
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing CSV: {str(e)}")

@router.post("/analyze-csv-network")
async def analyze_csv_network(
    file: UploadFile = File(...),
    source_column: str = Form(..., description="Name of source user column"),
    target_column: str = Form(..., description="Name of target user column"),
    interaction_type_column: Optional[str] = Form(None, description="Column for interaction type"),
    weight_column: Optional[str] = Form(None, description="Column for interaction weight"),
    top_n: int = Form(20, description="Number of top influencers to return")
):
    """
    Upload CSV and analyze network influence
    
    Expected CSV format:
    - source_column: User who initiated interaction
    - target_column: User who received interaction  
    - interaction_type_column: Type of interaction (optional)
    - weight_column: Weight/strength of interaction (optional)
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = [source_column, target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_cols}. Available: {list(df.columns)}"
            )
        
        # Prepare interaction data
        interaction_data = df[[source_column, target_column]].copy()
        interaction_data.columns = ['source_user', 'target_user']
        
        # Add interaction type
        if interaction_type_column and interaction_type_column in df.columns:
            interaction_data['interaction_type'] = df[interaction_type_column]
        else:
            interaction_data['interaction_type'] = 'interaction'
        
        # Add weight
        if weight_column and weight_column in df.columns:
            interaction_data['weight'] = pd.to_numeric(df[weight_column], errors='coerce').fillna(1.0)
        else:
            interaction_data['weight'] = 1.0
        
        # Remove rows with missing source/target
        interaction_data = interaction_data.dropna(subset=['source_user', 'target_user'])
        
        if len(interaction_data) == 0:
            raise HTTPException(status_code=400, detail="No valid interaction data found")
        
        # Initialize analyzer and build network
        analyzer = PoliticalSentimentAnalyzer()
        analyzer.build_interaction_network(interaction_data)
        analyzer.calculate_influence_metrics()
        
        # Get top influencers
        top_influencers = analyzer.identify_key_opinion_leaders(top_n)
        
        # Format response
        influencer_data = []
        for user, metrics in top_influencers:
            influencer_data.append({
                'user': user,
                'influence_score': metrics['total_influence'],
                'pagerank': metrics['pagerank'],
                'degree_centrality': metrics['degree_centrality'],
                'betweenness_centrality': metrics['betweenness_centrality'],
                'eigenvector_centrality': metrics['eigenvector_centrality'],
                'in_degree': metrics['in_degree'],
                'out_degree': metrics['out_degree']
            })
        
        # Network statistics
        import networkx as nx
        graph = analyzer.network_graph
        
        response = {
            "file_info": {
                "filename": file.filename,
                "total_interactions": len(df),
                "valid_interactions": len(interaction_data),
                "source_column": source_column,
                "target_column": target_column
            },
            "network_analysis": {
                "total_users": len(graph.nodes()),
                "total_connections": len(graph.edges()),
                "network_density": float(nx.density(graph)),
                "is_connected": nx.is_weakly_connected(graph),
                "number_of_components": len(list(nx.weakly_connected_components(graph))),
                "top_influencers": influencer_data
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing network CSV: {str(e)}")