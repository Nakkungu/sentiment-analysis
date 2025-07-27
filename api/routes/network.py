from fastapi import APIRouter, HTTPException
from app.models import NetworkAnalysisRequest, InfluenceResponse
from core.political_analyzer import PoliticalSentimentAnalyzer
import pandas as pd

router = APIRouter()
analyzer = PoliticalSentimentAnalyzer()

@router.post("/analyze-network", response_model=InfluenceResponse)
async def analyze_network(request: NetworkAnalysisRequest):
    """Analyze political influence network"""
    try:
        # Convert request data to DataFrame
        interactions_df = pd.DataFrame([
            {
                'source_user': interaction.source_user,
                'target_user': interaction.target_user,
                'interaction_type': interaction.interaction_type,
                'weight': interaction.weight
            }
            for interaction in request.interactions
        ])
        
        # Build network and calculate influence
        analyzer.build_interaction_network(interactions_df)
        analyzer.calculate_influence_metrics()
        
        # Get top influencers
        top_influencers = analyzer.identify_key_opinion_leaders(request.top_n or 20)
        
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
        
        return {
            'total_users': len(analyzer.network_graph.nodes()),
            'total_connections': len(analyzer.network_graph.edges()),
            'network_density': float(analyzer.network_graph.number_of_edges() / (len(analyzer.network_graph.nodes()) * (len(analyzer.network_graph.nodes()) - 1)) if len(analyzer.network_graph.nodes()) > 1 else 0),
            'top_influencers': influencer_data,
            'network_stats': {
                'is_connected': len(list(analyzer.network_graph.weakly_connected_components())) == 1,
                'components': len(list(analyzer.network_graph.weakly_connected_components()))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in network analysis: {str(e)}")