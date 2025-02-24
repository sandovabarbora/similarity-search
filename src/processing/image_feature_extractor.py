import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from typing import List, Union, Tuple, Optional
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """Dataset for parallel image loading"""
    def __init__(self, paths: List[Union[str, Path]], transform):
        self.paths = [Path(p) if isinstance(p, str) else p for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img_path = self.paths[idx]
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
            return tensor, str(img_path)
        except Exception as e:
            logger.error(f"Error loading image {self.paths[idx]}: {e}")
            return None, str(img_path)

class ImageFeatureExtractor:
    def __init__(self, num_workers: Optional[int] = None):
        logger.info("Initializing FeatureExtractor")
        try:
            # Determine optimal number of workers
            if num_workers is None:
                num_workers = min(os.cpu_count() or 1, 8)  # Limit to 8 workers
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
            if hasattr(torch, 'set_num_threads'):
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

    def process_batch(self, batch_tensors: torch.Tensor, batch_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Process a single batch of images"""
        valid_mask = torch.tensor([t is not None for t in batch_tensors])
        if not valid_mask.any():
            return np.array([]), []

        valid_tensors = torch.stack([t for t, m in zip(batch_tensors, valid_mask) if m])
        valid_paths = [p for p, m in zip(batch_paths, valid_mask) if m]

        with torch.no_grad():
            if self.device.type == "mps":
                tensors = valid_tensors.to(self.device, memory_format=torch.channels_last)
            else:
                tensors = valid_tensors.to(self.device)
            
            features = self.model(tensors)
            
        features = features.cpu().numpy()
        features = features.reshape(features.shape[0], -1)
        features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
        
        return features, valid_paths

    def batch_extract_features(self, 
                             image_paths: List[Union[str, Path]], 
                             batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
        """Extract features using efficient parallel processing"""
        try:
            logger.info("Starting parallel feature extraction")
            
            # Initialize dataset and dataloader
            dataset = ImageDataset(image_paths, self.transform)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

            all_features = []
            valid_paths = []
            total_batches = len(dataloader)

            # Process batches with progress tracking
            with tqdm(total=total_batches, desc="Extracting features") as pbar:
                for batch_tensors, batch_paths in dataloader:
                    features, paths = self.process_batch(batch_tensors, batch_paths)
                    
                    if len(features) > 0:
                        all_features.append(features)
                        valid_paths.extend(paths)

                    pbar.update(1)
                    
                    # Log progress periodically
                    if len(valid_paths) % (batch_size * 10) == 0:
                        logger.info(
                            f"Processed {len(valid_paths)}/{len(image_paths)} images"
                        )

            if not all_features:
                return np.array([]), []

            # Combine features
            final_features = np.vstack(all_features)
            
            logger.info(
                "Processing completed",
                extra={
                    'total_images': len(image_paths),
                    'processed_images': len(valid_paths),
                    'feature_shape': final_features.shape
                }
            )
            
            return final_features, valid_paths
            
        except Exception as e:
            logger.exception("Error during batch processing")
            raise

    def __del__(self):
        """Cleanup resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()