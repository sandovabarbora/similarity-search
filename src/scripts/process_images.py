import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from datetime import datetime
import time
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import os
import psutil
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import gc
from dataclasses import dataclass
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.logger import logger, Logger
from utils.log_decorators import log_function_call

@dataclass
class ProcessingStats:
    """Statistics for monitoring processing performance"""
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return (self.processed_images / self.total_images) * 100 if self.total_images > 0 else 0
    
    @property
    def images_per_second(self) -> float:
        return self.processed_images / self.processing_time if self.processing_time > 0 else 0

class ImageDataset(Dataset):
    """Dataset for parallel image loading with robust error handling"""
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
            logger.warning(f"Error loading image {self.paths[idx]}: {e}")
            return "LOAD_FAILED", str(img_path)

def custom_collate(batch):
    """Custom collate function that handles failed loads"""
    tensors = []
    paths = []
    
    for item, path in batch:
        if item != "LOAD_FAILED":
            tensors.append(item)
            paths.append(path)
    
    if not tensors:
        return torch.tensor([]), []
        
    return torch.stack(tensors), paths

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
        if len(batch_tensors) == 0:
            return np.array([]), []

        with torch.no_grad():
            if self.device.type == "mps":
                tensors = batch_tensors.to(self.device, memory_format=torch.channels_last)
            else:
                tensors = batch_tensors.to(self.device)
            
            features = self.model(tensors)
            
        features = features.cpu().numpy()
        features = features.reshape(features.shape[0], -1)
        features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
        
        return features, batch_paths

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
                persistent_workers=True,
                collate_fn=custom_collate
            )

            all_features = []
            valid_paths = []
            total_batches = len(dataloader)

            # Process batches with progress tracking
            with tqdm(total=total_batches, desc="Extracting features") as pbar:
                for batch_tensors, batch_paths in dataloader:
                    if len(batch_tensors) > 0:
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

def scan_chunk(chunk: List[Path], extensions: Set[str]) -> List[Path]:
    """Process a chunk of directories to find images"""
    images = []
    for path in chunk:
        try:
            if path.is_dir():
                for ext in extensions:
                    images.extend(path.glob(f"*.{ext.lower()}"))
                    images.extend(path.glob(f"*.{ext.upper()}"))
        except Exception as e:
            logger.error(f"Error scanning directory {path}: {e}")
    return images

class FastDirectoryScanner:
    """Optimized parallel directory scanner"""
    def __init__(self, extensions: Set[str]):
        self.extensions = extensions
    
    def parallel_scan(self, root_dir: Path, num_workers: int) -> List[Path]:
        """Parallel directory scanning with optimized memory usage"""
        try:
            # Get all subdirectories
            logger.debug("Getting subdirectories")
            subdirs = [root_dir] + [d for d in root_dir.rglob("*") if d.is_dir()]
            
            # Calculate optimal chunk size
            chunk_size = max(100, len(subdirs) // (num_workers * 4))
            chunks = [subdirs[i:i + chunk_size] 
                     for i in range(0, len(subdirs), chunk_size)]
            
            logger.info(f"Starting parallel scan with {num_workers} workers")
            all_images = []
            
            # Process in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for chunk in chunks:
                    futures.append(executor.submit(scan_chunk, chunk, self.extensions))
                
                # Collect results with progress tracking
                for future in tqdm(futures, desc="Scanning directories",
                                 total=len(chunks), unit="chunk"):
                    chunk_images = future.result()
                    all_images.extend(chunk_images)
                    
                    # Periodic memory cleanup
                    if len(all_images) % 10000 == 0:
                        gc.collect()
            
            # Remove duplicates and sort
            unique_images = sorted(set(str(p) for p in all_images))
            return [Path(p) for p in unique_images]
            
        except Exception as e:
            logger.exception("Error in parallel directory scan")
            raise

def get_system_stats() -> Dict[str, float]:
    """Get current system resource usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'memory_gb': memory_info.rss / (1024 * 1024 * 1024),
        'cpu_percent': process.cpu_percent(),
        'num_threads': process.num_threads()
    }

class H5Writer:
    """Efficient H5 file writer"""
    def __init__(self, filename: str):
        self.filename = filename

    def write(self, features: np.ndarray, paths: List[str], stats: ProcessingStats):
        """Write features and metadata to H5 file"""
        with h5py.File(self.filename, 'w') as h5_file:
            # Write features with optimal chunk size
            chunk_size = min(1000, len(features))
            h5_file.create_dataset(
                'features',
                data=features,
                chunks=(chunk_size, features.shape[1]),
                compression='gzip',
                compression_opts=1
            )

            # Write paths
            dt = h5py.special_dtype(vlen=str)
            paths_array = np.array(paths, dtype=dt)
            h5_file.create_dataset('paths', data=paths_array)

            # Write metadata
            system_stats = get_system_stats()
            metadata = {
                'num_images': len(paths),
                'feature_dim': features.shape[1],
                'creation_time': str(datetime.now()),
                'processing_time': stats.processing_time,
                'success_rate': stats.success_rate,
                'images_per_second': stats.images_per_second,
                'memory_usage_gb': system_stats['memory_gb'],
                'cpu_percent': system_stats['cpu_percent']
            }
            
            for key, value in metadata.items():
                h5_file.attrs[key] = value

@log_function_call
def process_image_dataset(
    input_dir: str,
    output_file: str,
    batch_size: int = 32,
    num_workers: Optional[int] = None
) -> ProcessingStats:
    """Process image dataset with optimized performance for M2"""
    stats = ProcessingStats()
    start_time = time.time()
    
    try:
        logger.info(f"Processing images from {input_dir}")
        input_path = Path(input_dir)
        
        # Scan for images using parallel processing
        scanner = FastDirectoryScanner({'jpg', 'jpeg', 'png'})
        num_workers = num_workers or min(os.cpu_count() or 1, 8)
        
        logger.info("Scanning for images...")
        image_paths = scanner.parallel_scan(input_path, num_workers)
        stats.total_images = len(image_paths)
        
        if stats.total_images == 0:
            raise ValueError(f"No images found in {input_dir}")
        
        logger.info(f"Found {stats.total_images} images")
        
        # Initialize feature extractor
        extractor = ImageFeatureExtractor(num_workers=num_workers)
        
        # Extract features
        logger.info("Extracting features...")
        features, valid_paths = extractor.batch_extract_features(
            image_paths,
            batch_size=batch_size
        )
        
        stats.processed_images = len(valid_paths)
        stats.failed_images = stats.total_images - stats.processed_images
        
        if stats.processed_images == 0:
            raise ValueError("Feature extraction failed for all images")
        
        # Save results
        logger.info(f"Saving features to {output_file}")
        writer = H5Writer(output_file)
        writer.write(features, valid_paths, stats)
        
        # Calculate final statistics
        stats.processing_time = time.time() - start_time
        system_stats = get_system_stats()
        stats.memory_usage = system_stats['memory_gb']
        stats.cpu_usage = system_stats['cpu_percent']
        
        logger.info(
            "Processing completed",
            extra={
                'total_time': f"{stats.processing_time:.2f}s",
                'images_per_second': f"{stats.images_per_second:.2f}",
                'success_rate': f"{stats.success_rate:.2f}%",
                'memory_usage_gb': f"{stats.memory_usage:.2f}",
                'cpu_usage': f"{stats.cpu_usage:.2f}%"
            }
        )
        
        return stats
        
    except Exception as e:
        logger.exception(f"Error processing dataset: {str(e)}")
        raise
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Process image dataset with optimized performance for M2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing images')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Output H5 file path')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--num_workers', type=int,
                      default=min(8, os.cpu_count() or 1),
                      help='Number of worker threads')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite output file if it exists')
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Directory not found: {args.input_dir}")
            
        output_file = Path(args.output_file)
        if output_file.exists() and not args.overwrite:
            raise ValueError(f"Output file exists: {args.output_file}. Use --overwrite to force.")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process dataset
        stats = process_image_dataset(
            args.input_dir,
            args.output_file,
            args.batch_size,
            args.num_workers
        )
        
        logger.info(
            "Script completed successfully",
            extra={
                'processed_images': stats.processed_images,
                'failed_images': stats.failed_images,
                'success_rate': f"{stats.success_rate:.2f}%",
                'processing_time': f"{stats.processing_time:.2f}s",
                'images_per_second': f"{stats.images_per_second:.2f}"
            }
        )
        
    except Exception as e:
        logger.exception(f"Script execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()