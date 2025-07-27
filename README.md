# Political Sentiment Analysis API

A comprehensive FastAPI application for political sentiment analysis, influence network mapping, and topic modeling.

## Features

- 📊 **Sentiment Analysis**: Analyze political texts for sentiment and polarity
- 🕸️ **Network Analysis**: Map influence networks and identify key opinion leaders  
- 🎯 **Topic Modeling**: Discover political themes and trends
- 📈 **Visualization**: Generate network graphs and sentiment charts
- 🚀 **Easy Deployment**: Support for Railway, Render, Fly.io, Heroku
- 📚 **Interactive Docs**: Auto-generated API documentation

## Quick Start

1. **Setup the project:**
   ```bash
   ./setup.sh
   ```

2. **Run the API:**
   ```bash
   ./run.sh
   ```

3. **Test the API:**
   ```bash
   python test_api.py
   ```

4. **View documentation:**
   Open http://localhost:8000/docs

## API Endpoints

- `POST /analyze-sentiment` - Analyze text sentiment
- `POST /analyze-network` - Analyze influence networks
- `POST /analyze-topics` - Topic modeling and analysis
- `POST /analyze-combined` - Combined analysis
- `POST /generate-visualization` - Create network visualizations
- `GET /health` - Health check

## Deployment

Deploy to your preferred platform:

```bash
# Railway (recommended)
./deploy.sh railway

# Other platforms
./deploy.sh render
./deploy.sh fly
./deploy.sh heroku
```

## Documentation

- [API Documentation](docs/API_DOCUMENTATION.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Data Format](docs/DATA_FORMAT.md)
- [Examples](docs/EXAMPLES.md)

## License

MIT License - see LICENSE file for details.
