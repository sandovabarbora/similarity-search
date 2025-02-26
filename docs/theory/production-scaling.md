# STRV Similarity Search: Production Deployment Guide

This guide provides detailed instructions for deploying STRV Similarity Search in a production environment, including hardware requirements, performance optimization, and scaling strategies.

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 100GB SSD (faster storage improves search performance)
- **Network**: 100Mbps minimum

### Recommended Configuration for Large Datasets
- **CPU**: 16+ cores
- **RAM**: 64GB+
- **GPU**: NVIDIA with 8GB+ VRAM for feature extraction
- **Storage**: 500GB+ NVMe SSD
- **Network**: 1Gbps+

## Deployment Options

### Docker Deployment (Recommended)

A Docker-based deployment provides the most consistent experience across environments.

#### Docker Compose Setup

```yaml
version: '3'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - NUM_WORKERS=4
      - VECTOR_DIMENSION=2048
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    restart: always
```

#### Dockerfiles

**Dockerfile.api**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Install FAISS for faster search
RUN pip install --no-cache-dir faiss-cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models

# Run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Dockerfile.frontend**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY frontend-requirements.txt .
RUN pip install --no-cache-dir -r frontend-requirements.txt

# Copy application code
COPY . .

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Bare Metal Deployment

For bare metal deployments, use a process manager like systemd:

**systemd service file for API (similarity-search-api.service)**:
```ini
[Unit]
Description=STRV Similarity Search API
After=network.target

[Service]
User=similarity
WorkingDirectory=/opt/similarity-search
ExecStart=/opt/similarity-search/env/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=on-failure
Environment=PATH=/opt/similarity-search/env/bin
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

**systemd service file for Frontend (similarity-search-frontend.service)**:
```ini
[Unit]
Description=STRV Similarity Search Frontend
After=network.target similarity-search-api.service

[Service]
User=similarity
WorkingDirectory=/opt/similarity-search
ExecStart=/opt/similarity-search/env/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
Environment=PATH=/opt/similarity-search/env/bin
Environment=PYTHONUNBUFFERED=1
Environment=API_URL=http://localhost:8000

[Install]
WantedBy=multi-user.target
```

## Storage Configuration

### Feature Database Management

The features.h5 file is critical for system performance. Recommendations:

1. **Storage Location**: Store on the fastest available storage
2. **Backup Strategy**: Create regular backups of the H5 file
3. **File System**: Use a file system that handles large files efficiently (ext4, XFS)
4. **Permissions**: Ensure the application has read permissions

### Image Storage

Options for storing the actual images:

1. **Local Storage**: Store images on local disk (simplest approach)
2. **NFS/Shared Storage**: Store on network-attached storage for multi-server setups
3. **Object Storage**: Use S3-compatible storage for cloud deployments

## Performance Optimization

### FAISS Optimization

If using FAISS for search acceleration:

1. **Index Type Selection**:
   - For datasets < 1M images: Use `IndexFlatIP` for exact search
   - For larger datasets: Use `IndexIVFFlat` or `IndexIVFPQ` for approximate search

2. **FAISS GPU Support**:
   If GPU acceleration is needed, install the GPU version:
   ```bash
   pip install faiss-gpu
   ```

3. **Index Tuning**:
   For `IndexIVF` variants, tune the number of clusters (nlist) based on dataset size:
   - Small datasets (< 100K): nlist = sqrt(n)
   - Medium datasets (100K-1M): nlist = 4 * sqrt(n)
   - Large datasets (> 1M): nlist = 16 * sqrt(n)

### API Performance

1. **Worker Configuration**:
   - Set workers to CPU cores - 1 for optimal performance
   - Example: On 8-core machine, use 7 workers

2. **Batch Size Tuning**:
   Adjust batch sizes based on available memory:
   - 8GB RAM: batch_size = 8
   - 16GB RAM: batch_size = 16
   - 32GB+ RAM: batch_size = 32

3. **Path Cache Optimization**:
   Ensure path lookup table is properly cached to avoid filesystem lookups

## Scaling Strategies

### Vertical Scaling

Optimize the application for larger machines:

1. **Memory Optimization**:
   - Increase batch sizes
   - Enable path caching
   - Tune Python garbage collection

2. **CPU Utilization**:
   - Increase worker count
   - Optimize thread allocation

3. **GPU Acceleration**:
   - Enable CUDA/MPS for feature extraction
   - Use FAISS-GPU for search acceleration

### Horizontal Scaling

For very large deployments, consider:

1. **Load Balancer + Multiple API Instances**:
   - Deploy multiple API instances behind a load balancer
   - Ensure all instances access the same feature database

2. **Database Scaling**:
   - Consider splitting the feature database into shards
   - Use a distributed vector database like Milvus for very large datasets

3. **Frontend Scaling**:
   - Deploy multiple frontend instances
   - Use sticky sessions if user state is important

## Security Considerations

1. **API Security**:
   - Add proper authentication (OAuth, API keys)
   - Rate limiting to prevent abuse
   - Input validation for all endpoints

2. **Network Security**:
   - Use HTTPS with proper certificates
   - Configure firewalls to restrict access
   - Use a reverse proxy like Nginx

3. **File Permissions**:
   - Restrict access to the feature database
   - Run services with minimal required permissions

## Monitoring and Maintenance

### Monitoring Setup

1. **Log Collection**:
   - Configure log rotation to prevent disk filling
   - Set up log aggregation (ELK Stack, Graylog)

2. **Performance Monitoring**:
   - Set up Prometheus + Grafana for metrics
   - Monitor key metrics:
     - API response time
     - Memory usage
     - Search latency
     - Feature extraction time

3. **Alerts**:
   - Configure alerts for:
     - High error rates
     - API unavailability
     - Excessive resource usage

### Maintenance Tasks

1. **Regular Updates**:
   - Update the feature database as new images are added
   - Schedule updates during low-traffic periods

2. **Backup Strategy**:
   - Regular backups of the feature database
   - Test restoration procedures

3. **Performance Tuning**:
   - Periodically review logs for performance bottlenecks
   - Adjust configuration based on changing usage patterns

## Example Deployment Checklist

- [ ] Verify system requirements
- [ ] Prepare features.h5 file with processed images
- [ ] Configure environment variables
- [ ] Set up Docker or systemd services
- [ ] Configure networking and firewall
- [ ] Implement monitoring and logging
- [ ] Test all endpoints
- [ ] Set up backup procedures
- [ ] Document deployment details

## Troubleshooting Common Issues

### API Fails to Start

**Symptoms**: The API service doesn't start or crashes immediately.

**Potential Solutions**:
1. Check logs for specific errors
2. Verify the features.h5 file exists and is accessible
3. Ensure all dependencies are installed
4. Check port availability

### Search Performance Issues

**Symptoms**: Search operations are slow or time out.

**Potential Solutions**:
1. Enable FAISS for faster search
2. Increase worker count
3. Move features.h5 to faster storage
4. Optimize batch sizes

### Path Resolution Failures

**Symptoms**: API returns errors about image paths not found.

**Potential Solutions**:
1. Check file permissions
2. Verify image paths in the H5 file
3. Ensure storage is properly mounted
4. Use the ensure_valid_path function

### Memory Usage Problems

**Symptoms**: Services use excessive memory or crash with OOM errors.

**Potential Solutions**:
1. Reduce batch sizes
2. Enable more aggressive garbage collection
3. Upgrade to a machine with more RAM
4. Use memory profiling to identify leaks