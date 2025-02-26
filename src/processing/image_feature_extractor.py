import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from typing import List, Union, Tuple, Optional
import numpy as np
from pathlib import Path
import os
import logging
import gc

logger = logging.getLogger(__name__)

class ImageFeatureExtractor:
    def __init__(self, num_workers: Optional[int] = None, max_batch_size: int = 64):
        """
        Initialize feature extractor with advanced configuration
        
        Args:
            num_workers (Optional[int]): Number of worker threads
            max_batch_size (int): Maximum batch size for processing
        """
        logger.info("Initializing Advanced FeatureExtractor")
        
        try:
            # Configure workers
            self.num_workers = num_workers or min(os.cpu_count() or 1, 8)
            self.max_batch_size = max_batch_size
            
            # Device selection with comprehensive fallback
            self.device = self._select_optimal_device()
            
            # Model and transform preparation
            self.model = self._prepare_model()
            self.transform = self._create_transform()
            
            logger.info(f"FeatureExtractor initialized on {self.device}")
        
        except Exception as e:
            logger.exception(f"Feature extractor initialization failed: {e}")
            raise

    def _select_optimal_device(self) -> torch.device:
        """
        Intelligently select the optimal device with detailed logging
        
        Returns:
            torch.device: Best available device
        """
        device_priority = [
            (torch.cuda.is_available(), "cuda"),
            (torch.backends.mps.is_available(), "mps"),
            (True, "cpu")
        ]
        
        for is_available, device_type in device_priority:
            if is_available:
                try:
                    device = torch.device(device_type)
                    
                    # Additional device-specific logging
                    if device_type == "cuda":
                        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                    
                    return device
                except Exception as e:
                    logger.warning(f"Could not use {device_type} device: {e}")
        
        raise RuntimeError("No suitable device found for feature extraction")

    def _prepare_model(self) -> torch.nn.Module:
        """
        Prepare and optimize the feature extraction model
        
        Returns:
            torch.nn.Module: Prepared feature extraction model
        """
        try:
            # Load pretrained ResNet50 and remove classification layer
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            # Device-specific optimizations
            if self.device.type == "mps":
                model = model.to(memory_format=torch.channels_last)
            
            # Move to selected device
            model = model.to(self.device)
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            return model
        
        except Exception as e:
            logger.exception(f"Model preparation failed: {e}")
            raise

    def _create_transform(self) -> transforms.Compose:
        """
        Create image transformation pipeline
        
        Returns:
            transforms.Compose: Image transformation pipeline
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_single_image_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from a single image with comprehensive error handling
        
        Args:
            image (Image.Image): Input image
        
        Returns:
            np.ndarray: Normalized feature vector
        """
        try:
            # Ensure cleanup between extractions
            torch.cuda.empty_cache()
            gc.collect()
            
            with torch.no_grad():
                # Prepare image tensor
                img_tensor = self.transform(image).unsqueeze(0)
                
                # Device-specific tensor transfer
                if self.device.type == "mps":
                    img_tensor = img_tensor.to(
                        self.device, 
                        memory_format=torch.channels_last
                    )
                elif self.device.type == "cuda":
                    img_tensor = img_tensor.pin_memory().to(
                        self.device, 
                        non_blocking=True
                    )
                else:
                    img_tensor = img_tensor.to(self.device)
                
                # Extract features
                features = self.model(img_tensor)
                
                # Transfer to CPU and convert to numpy
                features_np = features.cpu().numpy()
                features_np = features_np.reshape(features_np.shape[0], -1)
                
                # Normalize features
                features_norm = features_np / np.linalg.norm(features_np, axis=1)[:, np.newaxis]
                
                return features_norm[0]
        
        except RuntimeError as re:
            logger.error(f"Runtime error in feature extraction: {re}")
            if "out of memory" in str(re):
                torch.cuda.empty_cache()
            raise
        
        except Exception as e:
            logger.exception(f"Feature extraction failed: {e}")
            raise
        
        finally:
            # Ensure resource cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def batch_extract_features(
        self, 
        image_paths: List[Union[str, Path]], 
        batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple images with advanced batch processing
        
        Args:
            image_paths (List[Union[str, Path]]): Paths to images
            batch_size (Optional[int]): Batch size for processing
        
        Returns:
            Tuple[np.ndarray, List[str]]: Features and corresponding paths
        """
        # Validate and adjust batch size
        batch_size = batch_size or min(self.max_batch_size, len(image_paths))
        
        # Tracking variables
        all_features = []
        valid_paths = []
        
        logger.info(f"Starting batch feature extraction for {len(image_paths)} images")
        
        try:
            # Batch processing with granular error handling
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                
                try:
                    batch_features, batch_valid_paths = self._process_image_batch(batch_paths)
                    
                    if len(batch_features) > 0:
                        all_features.append(batch_features)
                        valid_paths.extend(batch_valid_paths)
                
                except Exception as batch_error:
                    logger.warning(f"Error processing batch {i//batch_size}: {batch_error}")
                
                # Periodic cleanup
                if (i + batch_size) % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Combine features
            if not all_features:
                logger.warning("No features extracted")
                return np.array([]), []
            
            final_features = np.vstack(all_features)
            
            logger.info(f"Extracted {len(valid_paths)} valid features")
            return final_features, valid_paths
        
        except Exception as e:
            logger.exception(f"Comprehensive feature extraction failed: {e}")
            raise
        
        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()

    def _process_image_batch(self, batch_paths: List[Union[str, Path]]) -> Tuple[np.ndarray, List[str]]:
        """
        Process a batch of images with robust error handling
        
        Args:
            batch_paths (List[Union[str, Path]]): Paths to images in the batch
        
        Returns:
            Tuple[np.ndarray, List[str]]: Extracted features and valid paths
        """
        batch_features = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                # Open and validate image
                image = Image.open(path).convert('RGB')
                
                # Extract features
                features = self.extract_single_image_features(image)
                
                batch_features.append(features)
                valid_paths.append(str(path))
            
            except Exception as img_error:
                logger.warning(f"Could not process image {path}: {img_error}")
                continue
        
        # Convert to numpy array if features exist
        if batch_features:
            return np.array(batch_features), valid_paths
        
        return np.array([]), []

    def compute_similarity(
        self, 
        query_features: np.ndarray, 
        feature_bank: np.ndarray, 
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute similarity between query features and feature bank
        
        Args:
            query_features (np.ndarray): Feature vector of the query image
            feature_bank (np.ndarray): Matrix of features to compare against
            top_k (int): Number of most similar results to return
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices and similarity scores of top-k results
        """
        try:
            # Normalize query features
            query_features = query_features.reshape(1, -1)
            query_features = query_features / np.linalg.norm(query_features, axis=1)[:, np.newaxis]
            
            # Compute cosine similarity
            similarities = np.dot(query_features, feature_bank.T)[0]
            
            # Get top-k indices and scores
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            return top_indices, top_scores
        
        except Exception as e:
            logger.exception(f"Similarity computation failed: {e}")
            raise

    def __del__(self):
        """Cleanup resources on object deletion"""
        try:
            # Ensure resources are freed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Additional cleanup
            del self.model
            gc.collect()
        except Exception as e:
            logger.error(f"Error during feature extractor cleanup: {e}")