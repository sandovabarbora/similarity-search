#!/usr/bin/env python3
import os
import sys
import logging
import traceback
import platform

# Comprehensive library and system diagnostics
def check_library_versions():
    """Check versions of key libraries"""
    libraries = [
        'numpy', 'torch', 'torchvision', 'faiss', 
        'h5py', 'PIL', 'fastapi', 'uvicorn'
    ]
    
    print("\n🔍 Library Versions:")
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            print(f"{lib}: {version}")
        except ImportError:
            print(f"{lib}: Not installed")
        except Exception as e:
            print(f"{lib}: Error checking version - {e}")

def system_diagnostics():
    """Collect system and environment information"""
    print("\n💻 System Diagnostics:")
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check CUDA availability
    try:
        import torch
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA Check Failed: {e}")

def validate_feature_file(h5_file_path):
    """
    Comprehensive validation of the feature H5 file
    
    Checks:
    - File existence
    - Dataset presence
    - Feature and path validation
    - Potential data integrity issues
    """
    import h5py
    import numpy as np
    
    print(f"\n🔬 Validating Feature File: {h5_file_path}")
    
    # Validate file existence
    if not os.path.exists(h5_file_path):
        print(f"❌ Error: File not found at {h5_file_path}")
        return False
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Check for required datasets
            required_datasets = ['features', 'paths']
            missing_datasets = [
                dataset for dataset in required_datasets 
                if dataset not in f.keys()
            ]
            
            if missing_datasets:
                print(f"❌ Missing datasets: {missing_datasets}")
                return False
            
            # Features analysis
            features = f['features']
            paths = f['paths']
            
            # Basic shape and dimension checks
            print(f"Total Images: {len(paths)}")
            print(f"Feature Dimensions: {features.shape}")
            
            # Detailed feature validation
            feature_array = features[:]
            
            # Check for NaN or Inf values
            nan_count = np.isnan(feature_array).sum()
            inf_count = np.isinf(feature_array).sum()
            
            print(f"NaN Values: {nan_count}")
            print(f"Inf Values: {inf_count}")
            
            # Feature statistics
            print("\nFeature Statistics:")
            print(f"Mean: {feature_array.mean()}")
            print(f"Std Dev: {feature_array.std()}")
            print(f"Min: {feature_array.min()}")
            print(f"Max: {feature_array.max()}")
            
            # Path validation
            valid_paths = sum(os.path.exists(path.decode('utf-8') if isinstance(path, bytes) else path) 
                              for path in paths[:10])  # Check first 10 paths
            print(f"Valid Paths (first 10): {valid_paths}/10")
            
            return True
    
    except Exception as e:
        print(f"❌ Error validating feature file: {e}")
        print(traceback.format_exc())
        return False

def check_faiss_compatibility():
    """
    Check FAISS library compatibility and basic functionality
    """
    print("\n🧩 FAISS Compatibility Check")
    
    try:
        import faiss
        import numpy as np
        
        print(f"FAISS Version: {faiss.__version__}")
        
        # Basic FAISS index creation test
        d = 64  # feature dimension
        nb = 1000  # database size
        
        # Generate random features
        np.random.seed(42)
        features = np.random.random((nb, d)).astype('float32')
        
        # Normalize features
        features /= np.linalg.norm(features, axis=1)[:, np.newaxis]
        
        # Create a simple index
        index = faiss.IndexFlatIP(d)
        index.add(features)
        
        print(f"Index created successfully")
        print(f"Total vectors in index: {index.ntotal}")
        
        # Simple search test
        k = 5  # top-k
        query = features[0].reshape(1, -1)
        D, I = index.search(query, k)
        
        print("Search test completed successfully")
        
    except ImportError:
        print("❌ FAISS is not installed")
    except Exception as e:
        print(f"❌ FAISS compatibility test failed: {e}")
        print(traceback.format_exc())

def main():
    """Main diagnostic script"""
    print("🔍 STRV Similarity Search Diagnostic Tool")
    
    # Default feature file path - adjust as needed
    h5_file_path = 'models/features.h5'
    
    # Run diagnostics
    check_library_versions()
    system_diagnostics()
    
    # Validate feature file
    validate_feature_file(h5_file_path)
    
    # Check FAISS compatibility
    check_faiss_compatibility()

if __name__ == '__main__':
    main()