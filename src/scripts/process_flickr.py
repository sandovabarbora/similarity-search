import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from datetime import datetime
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from processing.feature_extractor import FeatureExtractor
from utils.logger import logger, Logger
from utils.log_decorators import log_function_call

@log_function_call
def process_flickr30k(flickr_dir: str, output_file: str, batch_size: int = 32):
    """Process Flickr30k dataset and save features to H5 file."""
    # Create context-specific logger
    log_ctx = Logger().add_extra_fields(
        flickr_dir=flickr_dir,
        output_file=output_file,
        batch_size=batch_size
    )
    
    try:
        start_time = time.time()
        log_ctx.info("Starting Flickr30k dataset processing")
        
        # Initialize feature extractor
        log_ctx.debug("Initializing feature extractor")
        extractor = FeatureExtractor()
        
        # Get all image paths
        log_ctx.debug(f"Scanning directory: {flickr_dir}")
        image_paths = list(Path(flickr_dir).glob("*.jpg"))
        total_images = len(image_paths)
        log_ctx.info(f"Found {total_images} images")
        
        if total_images == 0:
            log_ctx.error("No images found in directory")
            raise ValueError(f"No .jpg files found in {flickr_dir}")
        
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
                'output_file_size': Path(output_file).stat().st_size / (1024 * 1024)  # Size in MB
            }
        )
        
    except Exception as e:
        log_ctx.exception(f"Error processing Flickr30k dataset: {str(e)}")
        raise

def main():
    """Main entry point with argument parsing and error handling."""
    logger.info("Starting Flickr30k processing script")
    
    try:
        parser = argparse.ArgumentParser(description='Process Flickr30k dataset')
        parser.add_argument('--flickr_dir', type=str, required=True,
                          help='Directory containing Flickr30k images')
        parser.add_argument('--output_file', type=str, required=True,
                          help='Output H5 file path')
        parser.add_argument('--batch_size', type=int, default=32,
                          help='Batch size for processing')
        
        args = parser.parse_args()
        
        # Validate arguments
        flickr_dir = Path(args.flickr_dir)
        if not flickr_dir.exists():
            raise ValueError(f"Directory not found: {args.flickr_dir}")
            
        output_dir = Path(args.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {args.batch_size}")
            
        # Process dataset
        process_flickr30k(args.flickr_dir, args.output_file, args.batch_size)
        
    except Exception as e:
        logger.exception(f"Script execution failed: {str(e)}")
        sys.exit(1)
        
    logger.info("Script completed successfully")

if __name__ == "__main__":
    main()