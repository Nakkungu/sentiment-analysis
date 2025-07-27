from fastapi import APIRouter
from core.political_analyzer import demo_political_analysis

router = APIRouter()

@router.get("/demo")
async def run_demo():
    """Run the political analysis demonstration"""
    # Capture the demo output
    analyzer = demo_political_analysis()
    return {"message": "Demo completed successfully", "status": "ready"}