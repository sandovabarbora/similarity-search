import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from typing import List, Union, Tuple
import numpy as np
from pathlib import Path

from utils.logger import logger, Logger
from utils.log_decorators import log_class_methods

@log_class_methods
class FeatureExtractor:
    def __init__(self):
        logger.info("Initializing FeatureExtractor")
        try:
            # Load pretrained ResNet50
            logger.debug("Loading pretrained ResNet50 model")
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            
            # Remove the final fully connected layer
            logger.debug("Removing final fully connected layer")
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            self.model = self.model.to(self.device)
            
            # Standard ImageNet transforms
            logger.debug("Setting up image transforms")
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            logger.info("FeatureExtractor initialized successfully")
            
        except Exception as e:
            logger.exception("Failed to initialize FeatureExtractor")
            raise

    def extract_features(self, img: Union[str, Path, Image.Image]) -> np.ndarray:
        """Extract features from a single image."""
        # Create context-specific logger
        log_ctx = Logger().add_extra_fields(
            image_type=type(img).__name__,
            image_path=str(img) if isinstance(img, (str, Path)) else "PIL_Image"
        )
        
        try:
            log_ctx.debug("Starting feature extraction")
            
            # Handle different input types
            if isinstance(img, (str, Path)):
                log_ctx.debug(f"Loading image from path: {img}")
                img = Image.open(img).convert('RGB')
                log_ctx.debug(f"Image loaded successfully: {img.size}")
            elif not isinstance(img, Image.Image):
                log_ctx.error(f"Invalid image type: {type(img)}")
                raise TypeError("Image must be PIL Image or path to image")

            # Transform and add batch dimension
            log_ctx.debug("Applying image transforms")
            img_tensor = self.transform(img).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # Extract features
            log_ctx.debug("Extracting features")
            with torch.no_grad():
                features = self.model(img_tensor)

            # Process features
            features = features.cpu().numpy().flatten()
            features = features / np.linalg.norm(features)
            
            log_ctx.info("Feature extraction completed successfully", 
                        extra={'feature_shape': features.shape})
            
            return features
            
        except Exception as e:
            log_ctx.exception(f"Error during feature extraction: {str(e)}")
            raise

    def batch_extract_features(self, 
                             image_paths: List[Union[str, Path]], 
                             batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """Extract features from a batch of images."""
        # Create context-specific logger
        log_ctx = Logger().add_extra_fields(
            total_images=len(image_paths),
            batch_size=batch_size
        )
        
        try:
            log_ctx.info("Starting batch feature extraction")
            all_features = []
            valid_paths = []

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                log_ctx.debug(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
                
                batch_tensors = []
                batch_valid_paths = []

                # Prepare batch
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        tensor = self.transform(img)
                        batch_tensors.append(tensor)
                        batch_valid_paths.append(img_path)
                    except Exception as e:
                        log_ctx.error(f"Error processing {img_path}: {str(e)}")
                        continue

                if not batch_tensors:
                    log_ctx.warning("No valid images in current batch")
                    continue

                # Process batch
                batch = torch.stack(batch_tensors).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.model(batch)

                # Process features
                features = features.cpu().numpy()
                features = features.reshape(features.shape[0], -1)
                features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]

                all_features.append(features)
                valid_paths.extend(batch_valid_paths)

                log_ctx.debug(f"Batch processed successfully, {len(batch_valid_paths)} images")

            if not all_features:
                log_ctx.warning("No features extracted from any images")
                return np.array([]), []

            final_features = np.vstack(all_features)
            log_ctx.info("Batch processing completed", 
                        extra={
                            'processed_images': len(valid_paths),
                            'failed_images': len(image_paths) - len(valid_paths),
                            'feature_shape': final_features.shape
                        })
            
            return final_features, valid_paths
            
        except Exception as e:
            log_ctx.exception("Error during batch processing")
            raise

    @staticmethod
    def compute_similarity(query_features: np.ndarray, 
                         database_features: np.ndarray, 
                         top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between query and database features.
        Returns indices and scores of top_k most similar images.
        """
        # Create context-specific logger
        log_ctx = Logger().add_extra_fields(
            query_shape=query_features.shape,
            database_shape=database_features.shape,
            top_k=top_k
        )
        
        try:
            log_ctx.debug("Computing similarities")
            
            # Compute cosine similarity
            similarities = np.dot(database_features, query_features)
            
            # Get top k indices and scores
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            log_ctx.info("Similarity computation completed", 
                        extra={
                            'max_similarity': float(top_scores[0]),
                            'min_similarity': float(top_scores[-1])
                        })
            
            return top_indices, top_scores
            
        except Exception as e:
            log_ctx.exception("Error computing similarities")
            raise