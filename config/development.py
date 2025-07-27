import os

class DevelopmentConfig:
    DEBUG = True
    TESTING = False
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # CORS
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]
    
    # Logging
    LOG_LEVEL = "DEBUG"
    
    # File Upload
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR = "data/uploads"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = False
