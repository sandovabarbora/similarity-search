import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from datetime import datetime
import time
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from processing.text_feature_extractor import TextFeatureExtractor
from utils.logger import logger, Logger
from utils.log_decorators import log_function_call

def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK resources: {str(e)}")

@log_function_call
def preprocess_text(text: str) -> str:
    """Advanced text preprocessing for tweets."""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Handle URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        
        # Handle mentions
        text = re.sub(r'@\w+', '[MENTION]', text)
        
        # Convert emoji to text
        text = emoji.demojize(text)
        
        # Handle hashtags - remove # but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle contractions
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return text

@log_function_call
def validate_tweet_dataframe(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has required columns and format."""
    required_columns = ['Tweet_ID', 'Text', 'Timestamp', 'Retweets', 'Likes']
    
    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return False
    
    # Check data types
    try:
        df['Tweet_ID'] = df['Tweet_ID'].astype(str)
        df['Text'] = df['Text'].astype(str)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Retweets'] = df['Retweets'].astype(int)
        df['Likes'] = df['Likes'].astype(int)
    except Exception as e:
        logger.error(f"Error converting data types: {str(e)}")
        return False
        
    return True

@log_function_call
@log_function_call
def process_twitter_dataset(input_file: str, output_file: str, batch_size: int = 32):
    """Process Twitter dataset and save features to H5 file."""
    log_ctx = Logger().add_extra_fields(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size
    )
    
    try:
        start_time = time.time()
        log_ctx.info("Starting Twitter dataset processing")
        
        # Initialize feature extractor
        log_ctx.debug("Initializing text feature extractor")
        extractor = TextFeatureExtractor()
        
        # Load and validate data
        log_ctx.debug(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file, low_memory=False)
        
        # Process data
        total_tweets = len(df)
        log_ctx.info(f"Loaded {total_tweets} tweets")
        
        if total_tweets == 0:
            log_ctx.error("No tweets found in file")
            raise ValueError(f"No tweets found in {input_file}")
        
        # Get processed tweet data
        tweet_data = extractor.process_tweet_data(df)
        
        # Preprocess texts
        log_ctx.info("Preprocessing tweets")
        texts = []
        valid_indices = []
        
        for idx, text in enumerate(tqdm(tweet_data['texts'])):
            try:
                processed_text = extractor.preprocess_tweet(text)
                if processed_text.strip():  # Check if text is not empty after preprocessing
                    texts.append(processed_text)
                    valid_indices.append(idx)
            except Exception as e:
                log_ctx.warning(f"Error processing tweet {idx}: {str(e)}")
                continue
        
        if not texts:
            raise ValueError("No valid texts after preprocessing")
        
        # Extract features
        log_ctx.info("Starting feature extraction")
        features = extractor.batch_extract_features(texts, batch_size=batch_size)
        
        # Verify feature shape
        if len(features.shape) != 2:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        log_ctx.debug(f"Feature shape: {features.shape}")
        
        # Save features and metadata
        log_ctx.debug(f"Saving features to {output_file}")
        try:
            with h5py.File(output_file, 'w') as f:
                # Save features
                f.create_dataset('features', data=features)
                
                # Save tweet information
                dt = h5py.special_dtype(vlen=str)
                
                # Save processed texts
                text_dataset = f.create_dataset('texts', (len(texts),), dtype=dt)
                text_dataset[:] = texts
                
                # Save tweet IDs
                tweet_ids = tweet_data['tweet_ids'][valid_indices]
                id_dataset = f.create_dataset('tweet_ids', (len(tweet_ids),), dtype=dt)
                id_dataset[:] = tweet_ids.astype(str)
                
                # Save engagement metrics
                f.create_dataset('retweets', data=tweet_data['retweets'][valid_indices])
                f.create_dataset('likes', data=tweet_data['likes'][valid_indices])
                
                # Save timestamps and usernames
                timestamps = tweet_data['timestamps'][valid_indices]
                time_dataset = f.create_dataset('timestamps', (len(timestamps),), dtype=dt)
                time_dataset[:] = [str(ts) for ts in timestamps]
                
                username_dataset = f.create_dataset('usernames', (len(valid_indices),), dtype=dt)
                username_dataset[:] = tweet_data['usernames'][valid_indices]
                
                # Save metadata
                metadata_grp = f.create_group('metadata')
                metadata_grp.attrs['num_tweets'] = len(texts)
                metadata_grp.attrs['feature_dim'] = features.shape[1]
                metadata_grp.attrs['creation_time'] = str(datetime.now())
                metadata_grp.attrs['processing_time'] = time.time() - start_time
                metadata_grp.attrs['batch_size'] = batch_size
                metadata_grp.attrs['input_file'] = input_file
                metadata_grp.attrs['total_input_tweets'] = total_tweets
                metadata_grp.attrs['processed_tweets'] = len(texts)
                
        except Exception as e:
            log_ctx.exception(f"Error saving H5 file: {str(e)}")
            raise
        
        # Log final statistics
        processing_time = time.time() - start_time
        log_ctx.info(
            "Processing completed successfully",
            extra={
                'total_time': round(processing_time, 2),
                'tweets_per_second': round(len(texts) / processing_time, 2),
                'processed_tweets': len(texts),
                'failed_tweets': total_tweets - len(texts),
                'output_file_path': output_file,
                'output_file_size': Path(output_file).stat().st_size / (1024 * 1024)  # Size in MB
            }
        )
        
    except Exception as e:
        log_ctx.exception(f"Error processing Twitter dataset: {str(e)}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process Twitter dataset and extract text features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input CSV file containing tweets'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output H5 file path for storing features'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )
    
    return parser.parse_args()

def main():
    """Main entry point with argument parsing and error handling."""
    logger.info("Starting Twitter dataset processing script")
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Validate input file
        input_file = Path(args.input_file)
        if not input_file.exists():
            raise ValueError(f"Input file not found: {args.input_file}")
            
        # Check output file
        output_file = Path(args.output_file)
        if output_file.exists() and not args.overwrite:
            raise ValueError(f"Output file already exists: {args.output_file}. Use --overwrite to force.")
            
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate batch size
        if args.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {args.batch_size}")
        
        # Process dataset
        process_twitter_dataset(
            str(input_file),
            str(output_file),
            args.batch_size
        )
        
    except Exception as e:
        logger.exception(f"Script execution failed: {str(e)}")
        sys.exit(1)
    
    logger.info("Script completed successfully")

if __name__ == "__main__":
    main()