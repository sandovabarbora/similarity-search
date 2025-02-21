import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from datetime import datetime
import time
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from processing.image_feature_extractor import ImageFeatureExtractor
from utils.logger import logger, Logger
from utils.log_decorators import log_function_call

def get_image_paths(directory: Path) -> List[Path]:
    """Get all jpg files from directory and its subdirectories."""
    try:
        # Get images from root directory
        root_images = list(directory.glob("*.jpg"))
        root_images.extend(directory.glob("*.JPG"))
        
        # Get images from subdirectories
        subdir_images = list(directory.rglob("*/*.jpg"))
        subdir_images.extend(directory.rglob("*/*.JPG"))
        
        # Combine and sort all paths
        all_images = root_images + subdir_images
        all_images.sort()
        
        return all_images
    except Exception as e:
        logger.exception(f"Error collecting image paths from {directory}")
        raise

def count_images_by_directory(image_paths: List[Path]) -> Dict[str, int]:
    """Count images in each directory."""
    counts = {}
    for path in image_paths:
        dir_path = str(path.parent)
        counts[dir_path] = counts.get(dir_path, 0) + 1
    return counts

@log_function_call
def process_image_dataset(input_dir: str, output_file: str, batch_size: int = 32):
    """Process image dataset and save features to H5 file."""
    log_ctx = Logger().add_extra_fields(
        input_dir=input_dir,
        output_file=output_file,
        batch_size=batch_size
    )
    
    try:
        start_time = time.time()
        log_ctx.info("Starting image dataset processing")
        
        # Initialize feature extractor
        log_ctx.debug("Initializing feature extractor")
        extractor = ImageFeatureExtractor()
        
        # Get all image paths
        log_ctx.debug(f"Scanning directory: {input_dir}")
        image_paths = get_image_paths(Path(input_dir))
        total_images = len(image_paths)
        
        # Count images by directory
        dir_counts = count_images_by_directory(image_paths)
        root_count = dir_counts.get(str(Path(input_dir)), 0)
        
        log_ctx.info(
            f"Found {total_images} images total",
            extra={
                'root_images': root_count,
                'subdirectory_images': total_images - root_count,
                'num_directories': len(dir_counts)
            }
        )
        
        if total_images == 0:
            log_ctx.error("No images found in directory")
            raise ValueError(f"No image files found in {input_dir} or its subdirectories")
        
        # Process images in batches
        log_ctx.info("Starting batch processing")
        features, valid_paths = extractor.batch_extract_features(
            image_paths,
            batch_size=batch_size
        )
        
        # Verify processing results
        if len(valid_paths) == 0:
            log_ctx.error("No valid features extracted")
            raise ValueError("Feature extraction failed for all images")
            
        log_ctx.info(
            "Feature extraction completed",
            extra={
                'processed_images': len(valid_paths),
                'failed_images': total_images - len(valid_paths),
                'feature_shape': features.shape
            }
        )
        
        # Save features and paths
        log_ctx.debug(f"Saving features to {output_file}")
        try:
            with h5py.File(output_file, 'w') as f:
                # Save features
                log_ctx.debug("Creating features dataset")
                f.create_dataset('features', data=features)
                
                # Save filenames as string attributes
                log_ctx.debug("Saving image paths")
                dt = h5py.special_dtype(vlen=str)
                path_dataset = f.create_dataset('image_paths', (len(valid_paths),), dtype=dt)
                path_dataset[:] = [str(p) for p in valid_paths]
                
                # Save metadata
                log_ctx.debug("Adding metadata")
                f.attrs['num_images'] = len(valid_paths)
                f.attrs['feature_dim'] = features.shape[1]
                f.attrs['creation_time'] = str(datetime.now())
                f.attrs['processing_time'] = time.time() - start_time
                f.attrs['batch_size'] = batch_size
                
                # Save directory information
                dir_info = f.create_group('directory_info')
                
                # Save directory counts
                dir_names = list(dir_counts.keys())
                dir_dataset = dir_info.create_dataset('directories', (len(dir_names),), dtype=dt)
                dir_dataset[:] = dir_names
                
                count_dataset = dir_info.create_dataset('counts', (len(dir_names),), dtype=np.int32)
                count_dataset[:] = [dir_counts[d] for d in dir_names]
                
        except Exception as e:
            log_ctx.exception(f"Error saving H5 file: {str(e)}")
            raise
            
        # Log final statistics
        processing_time = time.time() - start_time
        log_ctx.info(
            "Processing completed successfully",
            extra={
                'total_time': round(processing_time, 2),
                'images_per_second': round(len(valid_paths) / processing_time, 2),
                'output_file_path': output_file,
                'output_file_size': Path(output_file).stat().st_size / (1024 * 1024),  # Size in MB
                'root_directory': str(Path(input_dir)),
                'total_directories': len(dir_counts)
            }
        )
        
    except Exception as e:
        log_ctx.exception(f"Error processing image dataset: {str(e)}")
        raise

def main():
    """Main entry point with argument parsing and error handling."""
    logger.info("Starting image processing script")
    
    try:
        parser = argparse.ArgumentParser(
            description='Process image dataset from directory and subdirectories',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--input_dir', type=str, required=True,
                          help='Root directory containing images')
        parser.add_argument('--output_file', type=str, required=True,
                          help='Output H5 file path')
        parser.add_argument('--batch_size', type=int, default=32,
                          help='Batch size for processing')
        parser.add_argument('--overwrite', action='store_true',
                          help='Overwrite output file if it exists')
        
        args = parser.parse_args()
        
        # Validate arguments
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Directory not found: {args.input_dir}")
            
        output_file = Path(args.output_file)
        if output_file.exists() and not args.overwrite:
            raise ValueError(f"Output file already exists: {args.output_file}. Use --overwrite to force.")
            
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if args.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {args.batch_size}")
            
        # Process dataset
        process_image_dataset(args.input_dir, args.output_file, args.batch_size)
        
    except Exception as e:
        logger.exception(f"Script execution failed: {str(e)}")
        sys.exit(1)
        
    logger.info("Script completed successfully")

if __name__ == "__main__":
    main()