# Enhanced Storage Architecture for Social Media Platform

## 1. Vector Database Implementation (Milvus)

### Installation and Setup
```yaml
# docker-compose.yml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
```

### Python Implementation
```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Define collection schema
dim = 512  # ResNet50 feature dimension
collection_name = "image_features"

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="feature_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields=fields, description="Image feature vectors")
collection = Collection(name=collection_name, schema=schema)

# Create IVF_SQ8 index for fast retrieval
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_SQ8",
    "params": {"nlist": 1024}
}
collection.create_index(field_name="feature_vector", index_params=index_params)
```

## 2. Cassandra Implementation for Metadata

### Setup
```yaml
# docker-compose.cassandra.yml
version: '3'

services:
  cassandra:
    image: cassandra:latest
    ports:
      - "9042:9042"
    volumes:
      - cassandra_data:/var/lib/cassandra
    environment:
      - CASSANDRA_CLUSTER_NAME=ImageSocialCluster
      - CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch
      - CASSANDRA_DC=datacenter1

volumes:
  cassandra_data:
```

### Schema Definition
```sql
-- Keyspace creation
CREATE KEYSPACE IF NOT EXISTS social_images 
WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'datacenter1': 3
};

-- User uploads table
CREATE TABLE social_images.user_uploads (
    user_id uuid,
    upload_id uuid,
    upload_timestamp timestamp,
    image_path text,
    metadata map<text, text>,
    PRIMARY KEY ((user_id), upload_timestamp, upload_id)
) WITH CLUSTERING ORDER BY (upload_timestamp DESC);

-- Image metadata table
CREATE TABLE social_images.image_metadata (
    image_id uuid,
    user_id uuid,
    upload_timestamp timestamp,
    image_path text,
    feature_vector_id uuid,
    tags set<text>,
    metadata map<text, text>,
    PRIMARY KEY (image_id)
);

-- User interactions table
CREATE TABLE social_images.user_interactions (
    user_id uuid,
    interaction_date date,
    interaction_timestamp timestamp,
    image_id uuid,
    interaction_type text,
    PRIMARY KEY ((user_id, interaction_date), interaction_timestamp, image_id)
) WITH CLUSTERING ORDER BY (interaction_timestamp DESC);
```

### Python Implementation
```python
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from datetime import datetime
import uuid

class CassandraClient:
    def __init__(self, contact_points=['localhost'], port=9042):
        self.cluster = Cluster(contact_points, port=port)
        self.session = self.cluster.connect('social_images')
        self.prepare_statements()

    def prepare_statements(self):
        self.insert_upload = self.session.prepare("""
            INSERT INTO user_uploads 
            (user_id, upload_id, upload_timestamp, image_path, metadata)
            VALUES (?, ?, ?, ?, ?)
        """)
        
        self.insert_metadata = self.session.prepare("""
            INSERT INTO image_metadata 
            (image_id, user_id, upload_timestamp, image_path, feature_vector_id, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """)

    def save_upload(self, user_id, image_path, metadata, tags):
        upload_id = uuid.uuid4()
        feature_vector_id = uuid.uuid4()
        timestamp = datetime.utcnow()

        batch = BatchStatement()
        batch.add(self.insert_upload, 
                 (user_id, upload_id, timestamp, image_path, metadata))
        batch.add(self.insert_metadata,
                 (upload_id, user_id, timestamp, image_path, feature_vector_id, tags, metadata))
        
        self.session.execute(batch)
        return upload_id, feature_vector_id
```

## 3. Redis Cache Implementation

### Setup
```yaml
# docker-compose.redis.yml
version: '3'

services:
  redis:
    image: redis:latest
    command: redis-server --save 20 1 --loglevel warning
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Python Implementation
```python
import redis
from typing import List, Dict
import json

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)

    def cache_similar_images(self, query_id: str, similar_images: List[Dict],
                           expire_time: int = 3600):
        """Cache similar images results"""
        self.redis_client.setex(
            f"similar:{query_id}",
            expire_time,
            json.dumps(similar_images)
        )

    def get_cached_similar_images(self, query_id: str) -> List[Dict]:
        """Retrieve cached similar images"""
        cached = self.redis_client.get(f"similar:{query_id}")
        return json.loads(cached) if cached else None

    def cache_user_feed(self, user_id: str, feed_items: List[Dict],
                       expire_time: int = 300):
        """Cache user feed"""
        self.redis_client.setex(
            f"feed:{user_id}",
            expire_time,
            json.dumps(feed_items)
        )
```

## 4. Integration Example

```python
class StorageManager:
    def __init__(self):
        self.milvus_client = MilvusClient()
        self.cassandra_client = CassandraClient()
        self.redis_cache = RedisCache()

    async def save_image(self, user_id: uuid.UUID, image_path: str,
                        feature_vector: np.ndarray, metadata: Dict):
        # 1. Save to Cassandra
        upload_id, vector_id = self.cassandra_client.save_upload(
            user_id, image_path, metadata, tags=set()
        )

        # 2. Save feature vector to Milvus
        self.milvus_client.insert_vectors([{
            'id': vector_id,
            'image_path': image_path,
            'feature_vector': feature_vector
        }])

        return upload_id, vector_id

    async def find_similar_images(self, query_vector: np.ndarray,
                                top_k: int = 10) -> List[Dict]:
        # 1. Check cache
        cache_key = str(hash(query_vector.tobytes()))
        cached_results = self.redis_cache.get_cached_similar_images(cache_key)
        
        if cached_results:
            return cached_results

        # 2. Search in Milvus
        similar_vectors = self.milvus_client.search_vectors(
            query_vector, top_k=top_k
        )

        # 3. Get metadata from Cassandra
        results = self.cassandra_client.get_images_metadata(
            [v['id'] for v in similar_vectors]
        )

        # 4. Cache results
        self.redis_cache.cache_similar_images(cache_key, results)

        return results
```

## 5. Deployment Considerations

### Hardware Requirements
- Milvus: 
  - 32GB+ RAM for millions of vectors
  - SSD storage
  - Multiple CPU cores
- Cassandra:
  - Minimum 3 nodes for reliability
  - 16GB+ RAM per node
  - Fast storage for write operations
- Redis:
  - 8GB+ RAM
  - Optional persistence

### Scaling Strategy
1. **Horizontal Scaling**
   - Milvus: Add more query nodes
   - Cassandra: Add more nodes to the cluster
   - Redis: Implement Redis Cluster

2. **Monitoring**
   - Prometheus + Grafana for metrics
   - Alert manager for system health
   - Log aggregation (ELK Stack)

3. **Backup Strategy**
   - Regular Cassandra snapshots
   - Milvus metadata backups
   - Redis persistence configuration