import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Tuple
import numpy as np
from pathlib import Path

from src.utils.logger import logger, Logger
from src.utils.log_decorators import log_class_methods


class TextFeatureExtractor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the text feature extractor."""
        logger.info("Initializing TextFeatureExtractor")
        try:
            # Load pretrained transformer model
            logger.debug(f"Loading pretrained model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("TextFeatureExtractor initialized successfully")
            
        except Exception as e:
            logger.exception("Failed to initialize TextFeatureExtractor")
            raise

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on transformer outputs."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def extract_features(self, text: str) -> np.ndarray:
        """Extract features from a single text input."""
        log_ctx = Logger().add_extra_fields(text_length=len(text))
        
        try:
            log_ctx.debug("Starting feature extraction")
            
            # Tokenize and prepare input
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Extract features
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                features = self.mean_pooling(outputs, encoded_input['attention_mask'])

            # Process features
            features = features.cpu().numpy()
            features = features / np.linalg.norm(features)
            
            log_ctx.info("Feature extraction completed successfully", 
                        extra={'feature_shape': features.shape})
            
            return features[0]  # Return the first (and only) feature vector
            
        except Exception as e:
            log_ctx.exception(f"Error during feature extraction: {str(e)}")
            raise

    def batch_extract_features(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract features from a batch of texts."""
        log_ctx = Logger().add_extra_fields(
            total_texts=len(texts),
            batch_size=batch_size
        )
        
        try:
            log_ctx.info("Starting batch feature extraction")
            all_features = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Extract features
                with torch.no_grad():
                    outputs = self.model(**encoded_input)
                    features = self.mean_pooling(outputs, encoded_input['attention_mask'])

                # Process features
                features = features.cpu().numpy()
                features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
                all_features.append(features)

            final_features = np.vstack(all_features)
            log_ctx.info("Batch processing completed", 
                        extra={'feature_shape': final_features.shape})
            
            return final_features
            
        except Exception as e:
            log_ctx.exception("Error during batch processing")
            raise

    @staticmethod
    def compute_similarity(query_features: np.ndarray, 
                         database_features: np.ndarray, 
                         top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between query and database features.
        Returns indices and scores of top_k most similar texts.
        """
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