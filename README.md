# Plant Disease Detection System - Deployment Guide

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t plant-disease-app .
docker run -p 8501:8501 plant-disease-app
```

## 📋 Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Trained model file (*.pth)
- At least 2GB RAM recommended

## 🔧 Configuration

### Environment Variables
- `STREAMLIT_SERVER_HEADLESS=true` - Run in headless mode
- `PYTHONUNBUFFERED=1` - Ensure Python output is not buffered

### Model Files
Place your trained model files in the root directory with one of these names:
- `best_plant_disease_model.pth`
- `plant_disease_model_complete.pth` 
- `potato_disease_model.pth`

### Class Names
Optionally create `class_names.json` with your custom class names:
```json
["Class1", "Class2", "..."]
```

## 🌐 Deployment Options

### 1. Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

### 2. Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

### 3. AWS/GCP/Azure
- Use Docker container with container services
- Configure load balancing and auto-scaling as needed

### 4. Local Server
```bash
# Run with custom port
streamlit run app.py --server.port=8080

# Run in background
nohup streamlit run app.py &
```

## 📊 Performance Optimization

### Memory Management
- Model uses CPU-only PyTorch for broader compatibility
- Automatic garbage collection after predictions
- Image validation to prevent memory issues

### Caching
- Model loading is cached with `@st.cache_resource`
- Class names are cached for performance

### Resource Limits
- Docker container limited to 4GB RAM
- Image uploads limited to 10MB
- Image dimensions validated

## 🔒 Security Features

- Input validation for uploaded images
- File size and type restrictions
- CORS and XSRF protection configured
- Error handling prevents information leakage

## 📝 Monitoring

### Health Checks
- Docker health check endpoint: `/_stcore/health`
- Automatic container restart on failure

### Logging
- Structured logging with different levels
- Error tracking and debugging information
- Application performance monitoring

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure model file is in root directory
   - Check file naming convention

2. **Memory issues**
   - Reduce image size before upload
   - Increase Docker memory limits

3. **Port conflicts**
   - Change port in docker-compose.yml
   - Use different port with `--server.port`

### Debug Mode
```bash
# Run with debug logging
streamlit run app.py --logger.level=debug
```

## 📈 Scaling

### Horizontal Scaling
- Use multiple container instances
- Configure load balancer (nginx, HAProxy)
- Implement session affinity if needed

### Vertical Scaling
- Increase container memory/CPU limits
- Use GPU-enabled containers if available
- Optimize model inference batch size

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and Deploy
        run: |
          docker build -t plant-disease-app .
          # Add your deployment commands here
```

## 📋 Maintenance

### Regular Updates
- Update dependencies monthly
- Monitor security vulnerabilities
- Backup model files regularly

### Performance Monitoring
- Track response times
- Monitor memory usage
- Log prediction accuracy

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review application logs
3. Validate model and class files
4. Test with sample images

## 📄 License

Make sure to comply with licensing requirements for:
- PyTorch and torchvision
- Streamlit
- Any pre-trained models used