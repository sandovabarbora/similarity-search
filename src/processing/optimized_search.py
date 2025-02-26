import torch
import numpy as np
import h5py
import logging
from PIL import Image
import io
from typing import List, Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

class MPSOptimizedImageSearch:
    def __init__(self, h5_file_path: str):
        """
        Initialize MPS-optimized image search with robust error handling
        
        Args:
            h5_file_path (str): Path to the HDF5 feature file
        """
        self.h5_file_path = h5_file_path
        self.features = None
        self.paths = None
        
        try:
            self._load_features()
        except Exception as e:
            logger.error(f"Feature loading failed: {e}")
            raise

    def _load_features(self):
        """
        Load features with comprehensive error handling and memory optimization
        """
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                # Use float32 for memory efficiency
                self.features = f['features'][:].astype(np.float32)
                self.paths = f['paths'][:]
                
                # Normalize features during loading
                feature_norms = np.linalg.norm(self.features, axis=1)
                feature_norms[feature_norms == 0] = 1  # Avoid division by zero
                self.features /= feature_norms[:, np.newaxis]
                
                logger.info(f"Loaded {len(self.paths)} image features")
                logger.info(f"Feature matrix shape: {self.features.shape}")
        
        except Exception as e:
            logger.exception(f"Error loading features: {e}")
            raise

    def search_similar_images(
        self, 
        image_bytes: bytes, 
        top_k: int = 9, 
        similarity_threshold: float = 0.5  # Lowered threshold
    ) -> List[Dict[str, Any]]:
        """
        Find similar images with MPS-optimized feature extraction
        
        Args:
            image_bytes (bytes): Image file as bytes
            top_k (int): Number of top similar images to return
            similarity_threshold (float): Minimum similarity score
        
        Returns:
            List[Dict[str, Any]]: Similar image results
        """
        try:
            # Load and transform image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Extract features using MPS-optimized method
            query_features = self._extract_mps_features(image)
            
            # Detailed logging for feature extraction
            logger.info(f"Query feature vector: {query_features}")
            logger.info(f"Query feature norm: {np.linalg.norm(query_features)}")
            
            # Compute similarities with enhanced error handling
            try:
                # Ensure query_features is a 2D array for dot product
                query_features = query_features.reshape(1, -1)
                
                # Compute similarities
                similarities = np.dot(query_features, self.features.T)[0]
                
                # Log similarity statistics
                logger.info(f"Similarity range: [{np.min(similarities)}, {np.max(similarities)}]")
                logger.info(f"Mean similarity: {np.mean(similarities)}")
                logger.info(f"Median similarity: {np.median(similarities)}")
            
            except Exception as sim_error:
                logger.error(f"Similarity computation error: {sim_error}")
                return []
            
            # Get top-k results regardless of threshold
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Prepare results
            results = []
            for idx in top_indices:
                try:
                    # Handle potential byte/string path conversion
                    path = self.paths[idx]
                    if isinstance(path, bytes):
                        path = path.decode('utf-8')
                    
                    # Only include results above a minimal threshold
                    if similarities[idx] > similarity_threshold:
                        results.append({
                            'path': path,
                            'similarity_score': float(similarities[idx])
                        })
                except Exception as result_error:
                    logger.warning(f"Error processing result {idx}: {result_error}")
            
            # Log final results
            logger.info(f"Found {len(results)} similar images")
            for result in results:
                logger.info(f"Similar image: {result['path']}, Similarity: {result['similarity_score']:.4f}")
            
            return results
        
        except Exception as e:
            logger.exception(f"Comprehensive search failed: {e}")
            raise

    def _extract_mps_features(self, image: Image.Image) -> np.ndarray:
        """
        MPS-optimized feature extraction with comprehensive error handling
        
        Args:
            image (Image.Image): Input image
        
        Returns:
            np.ndarray: Normalized feature vector
        """
        try:
            # Ensure MPS is available
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS not available")
            
            # Import here to avoid global import issues
            import torchvision.transforms as transforms
            import torchvision.models as models
            
            # Prepare device and model
            device = torch.device('mps')
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model = model.to(device).eval()
            
            # Prepare transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Extract features
            with torch.no_grad():
                # Prepare tensor
                img_tensor = transform(image).unsqueeze(0)
                img_tensor = img_tensor.to(device, memory_format=torch.channels_last)
                
                # Extract features
                features = model(img_tensor)
                
                # Convert to numpy and normalize
                features_np = features.cpu().numpy()
                features_np = features_np.reshape(features_np.shape[0], -1)
                
                # Robust normalization
                feature_norm = np.linalg.norm(features_np, axis=1)
                if feature_norm[0] == 0:
                    logger.warning("Zero-norm feature vector detected")
                    feature_norm[feature_norm == 0] = 1.0
                
                features_norm = features_np / feature_norm[:, np.newaxis]
                
                return features_norm[0]
        
        except Exception as e:
            logger.exception(f"MPS feature extraction failed: {e}")
            raise

# Global search instance (can be initialized once and reused)
_global_search = None

def initialize_mps_search(h5_file_path: str):
    """
    Initialize global search instance
    
    Args:
        h5_file_path (str): Path to feature file
    """
    global _global_search
    _global_search = MPSOptimizedImageSearch(h5_file_path)

def mps_search_similar_images(image_bytes: bytes, top_k: int = 9) -> List[Dict[str, Any]]:
    """
    Global search function
    
    Args:
        image_bytes (bytes): Image file as bytes
        top_k (int): Number of top similar images to return
    
    Returns:
        List[Dict[str, Any]]: Similar image results
    """
    global _global_search
    if _global_search is None:
        raise RuntimeError("Search not initialized. Call initialize_mps_search first.")
    
    return _global_search.search_similar_images(image_bytes, top_k)