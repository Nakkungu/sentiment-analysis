from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import route modules
from api.routes import sentiment, network, topics, upload

# Create FastAPI app
app = FastAPI(
    title="Political Sentiment Analysis API",
    description="API for political sentiment analysis, influence network mapping, and topic modeling",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
origins = os.getenv("CORS_ORIGINS", "['*']")
app.add_middleware(
    CORSMiddleware,
    allow_origins=eval(origins) if isinstance(origins, str) else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sentiment.router, prefix="/api/v1", tags=["sentiment"])
app.include_router(network.router, prefix="/api/v1", tags=["network"])
app.include_router(topics.router, prefix="/api/v1", tags=["topics"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Political Sentiment Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "political-sentiment-api"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
