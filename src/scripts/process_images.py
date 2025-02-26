# Standard library imports
import sys
from pathlib import Path
import argparse
from datetime import datetime
import time
import os
import json
import platform
import gc
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass

# Scientific computing and data handling
import numpy as np
import h5py
import psutil

# Image and machine learning libraries
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torchvision.transforms as transforms
import torchvision.models as models  # Added this import
from torchvision.models import ResNet50_Weights  # Added for type hints
from PIL import Image

# Progress tracking
from tqdm import tqdm

# Temporary replacement for deprecated pkg_resources
try:
    import importlib.metadata as pkg_resources
except ImportError:
    import pkg_resources

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Project-specific imports
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

class H5FileManager:
    """Advanced H5 file management with append and update capabilities"""
    def __init__(self, filename: str, mode: str = 'a'):
        """
        Initialize H5 file manager
        
        Args:
            filename (str): Path to the H5 file
            mode (str, optional): File open mode. Defaults to 'a' (read/write/create)
        """
        self.filename = filename
        self.mode = mode

    def _convert_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert metadata to JSON-serializable format
        
        Args:
            metadata (Dict[str, Any]): Original metadata dictionary
        
        Returns:
            Dict[str, str]: Metadata with complex objects converted to JSON strings
        """
        converted_metadata = {}
        for key, value in metadata.items():
            try:
                # Try to store as-is first
                json.dumps(value)
                converted_metadata[key] = value
            except TypeError:
                # If JSON serialization fails, convert to string
                try:
                    converted_metadata[key] = json.dumps(value)
                except:
                    # Fallback to str representation if JSON fails
                    converted_metadata[key] = str(value)
        return converted_metadata

    def append_features(self, features: np.ndarray, paths: List[str], metadata: Dict[str, Any] = None):
        """
        Append features to an existing H5 file or create a new one
        
        Args:
            features (np.ndarray): Feature matrix to append
            paths (List[str]): Corresponding image paths
            metadata (Dict[str, Any], optional): Additional metadata to store
        """
        with h5py.File(self.filename, self.mode) as h5_file:
            # Append features
            if 'features' in h5_file:
                # Existing features dataset
                existing_features = h5_file['features']
                existing_paths = h5_file['paths']
                
                # Resize and append
                new_feature_shape = (existing_features.shape[0] + features.shape[0], features.shape[1])
                existing_features.resize(new_feature_shape)
                existing_features[-features.shape[0]:] = features
                
                # Append paths
                existing_paths.resize((existing_paths.shape[0] + len(paths),))
                existing_paths[-len(paths):] = paths
            else:
                # Create new datasets
                chunk_size = min(1000, len(features))
                h5_file.create_dataset(
                    'features', 
                    data=features, 
                    chunks=(chunk_size, features.shape[1]),
                    compression='gzip', 
                    compression_opts=1,
                    maxshape=(None, features.shape[1])
                )
                
                # Create paths dataset with string variable length
                dt = h5py.special_dtype(vlen=str)
                h5_file.create_dataset(
                    'paths', 
                    data=paths, 
                    dtype=dt,
                    maxshape=(None,)
                )
            
            # Update or create metadata
            if metadata:
                # Convert metadata to a JSON-serializable format
                converted_metadata = self._convert_metadata(metadata)
                
                # Store metadata as attributes
                for key, value in converted_metadata.items():
                    try:
                        h5_file.attrs[key] = value
                    except Exception as e:
                        logger.warning(f"Could not store metadata key '{key}': {e}")

    def get_library_versions(self) -> Dict[str, str]:
        """
        Retrieve versions of key libraries
        
        Returns:
            Dict[str, str]: Dictionary of library versions
        """
        libraries = [
            'numpy', 'torch', 'torchvision', 'h5py', 
            'tqdm', 'psutil', 'Pillow'
        ]
        
        versions = {}
        for lib in libraries:
            try:
                versions[lib] = pkg_resources.version(lib)
            except Exception:
                versions[lib] = 'Not found'
        
        return versions

class FastDirectoryScanner:
    """Optimized parallel directory scanner with configurable extensions"""
    def __init__(self, extensions: Optional[Set[str]] = None):
        """
        Initialize scanner with configurable image extensions
        
        Args:
            extensions (Optional[Set[str]]): Set of image extensions to scan. 
                                            Defaults to common image formats if not provided.
        """
        self.extensions = extensions or {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
    
    def parallel_scan(self, root_dir: Path, num_workers: int) -> List[Path]:
        """
        Parallel directory scanning with optimized memory usage and configurable extensions
        
        Args:
            root_dir (Path): Root directory to scan
            num_workers (int): Number of parallel workers
        
        Returns:
            List[Path]: Sorted list of unique image paths
        """
        try:
            # Get all subdirectories
            logger.debug("Getting subdirectories")
            subdirs = [root_dir] + [d for d in root_dir.rglob("*") if d.is_dir()]
            
            # Calculate optimal chunk size
            chunk_size = max(100, len(subdirs) // (num_workers * 4))
            chunks = [subdirs[i:i + chunk_size] 
                     for i in range(0, len(subdirs), chunk_size)]
            
            logger.info(f"Starting parallel scan with {num_workers} workers")
            logger.info(f"Scanning for extensions: {self.extensions}")
            
            all_images = []
            
            # Process in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for chunk in chunks:
                    futures.append(executor.submit(self._scan_chunk, chunk, self.extensions))
                
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

    @staticmethod
    def _scan_chunk(chunk: List[Path], extensions: Set[str]) -> List[Path]:
        """
        Process a chunk of directories to find images
        
        Args:
            chunk (List[Path]): List of directories to scan
            extensions (Set[str]): Image file extensions to find
        
        Returns:
            List[Path]: List of image file paths
        """
        images = []
        for path in chunk:
            try:
                if path.is_dir():
                    for ext in extensions:
                        # Scan for both lowercase and uppercase extensions
                        images.extend(path.glob(f"*.{ext.lower()}"))
                        images.extend(path.glob(f"*.{ext.upper()}"))
            except Exception as e:
                logger.error(f"Error scanning directory {path}: {e}")
        return images

class ImageDataset(torch.utils.data.Dataset):
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
            dataloader = torch.utils.data.DataLoader(
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

            # Process batches with a single progress bar
            with tqdm(total=total_batches, 
                    desc="Extracting Features", 
                    unit="batch", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for batch_tensors, batch_paths in dataloader:
                    if len(batch_tensors) > 0:
                        features, paths = self.process_batch(batch_tensors, batch_paths)
                        if len(features) > 0:
                            all_features.append(features)
                            valid_paths.extend(paths)

                    pbar.update(1)
                    pbar.set_postfix({
                        'processed_images': len(valid_paths),
                        'batch_size': len(batch_paths)
                    })

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

def get_system_stats() -> Dict[str, float]:
    """Get current system resource usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'memory_gb': memory_info.rss / (1024 * 1024 * 1024),
        'cpu_percent': process.cpu_percent(),
        'num_threads': process.num_threads()
    }

def process_image_dataset(
    input_dir: str,
    output_file: str,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    image_extensions: Optional[Set[str]] = None,
    append: bool = False
) -> Dict[str, Any]:
    """
    Process image dataset with advanced file management and configuration
    
    Args:
        input_dir (str): Input directory containing images
        output_file (str): Output H5 file path
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        num_workers (Optional[int], optional): Number of worker threads. Defaults to None.
        image_extensions (Optional[Set[str]], optional): Image file extensions to process. Defaults to None.
        append (bool, optional): Whether to append to existing file. Defaults to False.
    
    Returns:
        Dict[str, Any]: Processing statistics and metadata
    """
    # Initial setup
    stats = ProcessingStats()
    start_time = time.time()
    h5_manager = H5FileManager(output_file, mode='a' if append else 'w')
    
    try:
        logger.info(f"Processing images from {input_dir}")
        input_path = Path(input_dir)
        
        # Check existing paths if appending
        existing_paths = set()
        if append and os.path.exists(output_file):
            with h5py.File(output_file, 'r') as h5_file:
                if 'paths' in h5_file:
                    existing_paths = set(h5_file['paths'][:])
                    logger.info(f"Found {len(existing_paths)} existing images in the feature file")
        
        # Scan for images using parallel processing
        scanner = FastDirectoryScanner(image_extensions)
        num_workers = num_workers or min(os.cpu_count() or 1, 8)
        
        logger.info("Scanning for images...")
        image_paths = scanner.parallel_scan(input_path, num_workers)
        stats.total_images = len(image_paths)
        
        # Filter out already processed images
        new_image_paths = [path for path in image_paths if str(path) not in existing_paths]
        
        if not new_image_paths:
            logger.info("No new images to process.")
            return {}
        
        logger.info(f"Found {stats.total_images} total images")
        logger.info(f"Found {len(new_image_paths)} new images to process")
        
        # Update total images to new images count
        stats.total_images = len(new_image_paths)
        
        # Initialize feature extractor
        extractor = ImageFeatureExtractor(num_workers=num_workers)
        
        # Extract features
        logger.info("Extracting features...")
        features, valid_paths = extractor.batch_extract_features(
            new_image_paths,
            batch_size=batch_size
        )
        
        stats.processed_images = len(valid_paths)
        stats.failed_images = stats.total_images - stats.processed_images
        
        if stats.processed_images == 0:
            raise ValueError("Feature extraction failed for all images")
        
        # Prepare comprehensive metadata
        system_stats = get_system_stats()
        metadata = {
            'total_images_scanned': len(image_paths),
            'new_images_processed': len(valid_paths),
            'existing_images': len(existing_paths),
            'num_images': len(valid_paths),
            'feature_dim': features.shape[1],
            'creation_time': str(datetime.now()),
            'processing_time': time.time() - start_time,
            'success_rate': stats.success_rate,
            'images_per_second': stats.images_per_second,
            'memory_usage_gb': system_stats['memory_gb'],
            'cpu_percent': system_stats['cpu_percent'],
            'system_info': {
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'library_versions': h5_manager.get_library_versions()
            },
            'input_directory': str(input_path),
            'command_line_args': sys.argv
        }
        
        # Save results with advanced metadata
        logger.info(f"{'Appending' if append else 'Saving'} {len(valid_paths)} new features to {output_file}")
        h5_manager.append_features(features, valid_paths, metadata)
        
        # Calculate final statistics
        stats.processing_time = time.time() - start_time
        system_stats = get_system_stats()
        stats.memory_usage = system_stats['memory_gb']
        stats.cpu_usage = system_stats['cpu_percent']
        
        logger.info(
            "Processing completed",
            extra={
                'total_images_scanned': len(image_paths),
                'new_images_processed': len(valid_paths),
                'total_time': f"{stats.processing_time:.2f}s",
                'images_per_second': f"{stats.images_per_second:.2f}",
                'success_rate': f"{stats.success_rate:.2f}%",
                'memory_usage_gb': f"{stats.memory_usage:.2f}",
                'cpu_usage': f"{stats.cpu_usage:.2f}%"
            }
        )
        
        return metadata
        
    except Exception as e:
        logger.exception(f"Error processing dataset: {str(e)}")
        raise
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main entry point with enhanced argument parsing"""
    parser = argparse.ArgumentParser(
        description='Advanced Image Feature Extraction with Flexible Configuration',
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
    parser.add_argument('--extensions', type=str, nargs='+',
                      default=['jpg', 'jpeg', 'png'],
                      help='Image file extensions to process')
    parser.add_argument('--append', action='store_true',
                      help='Append to existing H5 file instead of overwriting')
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Directory not found: {args.input_dir}")
            
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process dataset
        metadata = process_image_dataset(
            args.input_dir,
            args.output_file,
            args.batch_size,
            args.num_workers,
            set(args.extensions),
            args.append
        )
        
        # Print metadata for user reference
        print(json.dumps(metadata, indent=2))
        
    except Exception as e:
        logger.exception(f"Script execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()