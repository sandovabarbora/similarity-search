import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


class ImageFeatureExtractor:
    def __init__(self, num_workers: Optional[int] = None):
        logger.info("Initializing FeatureExtractor")
        try:
            # Determine optimal number of workers
            if num_workers is None:
                num_workers = min(os.cpu_count() or 1, 8)
            self.num_workers = num_workers

            # Setup device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS backend")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA backend")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU backend")

            # Enable PyTorch optimizations
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(num_workers)

            # Load and optimize model
            logger.debug("Loading pretrained ResNet50 model")
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

            # Optimize for MPS if available
            if self.device.type == "mps":
                self.model = self.model.to(memory_format=torch.channels_last)

            self.model = self.model.to(self.device)
            self.model.eval()

            # Setup transforms
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            logger.info("FeatureExtractor initialized successfully")

        except Exception:
            logger.exception("Failed to initialize FeatureExtractor")
            raise

    def extract_single_image_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from a single image"""
        try:
            with torch.no_grad():
                # Transform and prepare image
                img_tensor = self.transform(image).unsqueeze(0)
                if self.device.type == "mps":
                    img_tensor = img_tensor.to(self.device, memory_format=torch.channels_last)
                else:
                    img_tensor = img_tensor.to(self.device)

                # Extract features
                features = self.model(img_tensor)
                features = features.cpu().numpy()
                features = features.reshape(features.shape[0], -1)
                features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]

                return features[0]  # Return the single feature vector

        except Exception:
            logger.exception("Error extracting features from single image")
            raise

    def compute_similarity(
        self, query_features: np.ndarray, feature_bank: np.ndarray, top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute similarity between query features and feature bank

        Args:
            query_features: Feature vector of the query image
            feature_bank: Matrix of features to compare against
            top_k: Number of most similar results to return

        Returns:
            Tuple of (indices, similarity_scores) for top-k most similar images
        """
        try:
            # Normalize query features if not already normalized
            query_features = query_features.reshape(1, -1)
            query_features = query_features / np.linalg.norm(query_features, axis=1)[:, np.newaxis]

            # Compute cosine similarity
            similarities = np.dot(query_features, feature_bank.T)[0]

            # Get top-k indices and scores
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]

            return top_indices, top_scores

        except Exception:
            logger.exception("Error computing similarity")
            raise

    def __del__(self):
        """Cleanup resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
