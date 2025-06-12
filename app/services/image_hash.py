"""
Perceptual image hashing for exact and near-exact duplicate detection.
Uses multiple hashing algorithms for robust duplicate detection.
"""

import hashlib
import numpy as np
import structlog
from PIL import Image
from typing import Tuple, Optional
import cv2

logger = structlog.get_logger()

def dhash(image_path: str, hash_size: int = 8) -> str:
    """
    Generate difference hash (dHash) for an image.
    Good for detecting duplicates with minor modifications.
    """
    try:
        # Load and convert image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Resize to hash_size + 1 x hash_size
        image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(image)
        
        # Calculate horizontal gradient
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # Convert boolean array to hash string
        hash_bits = ''.join('1' if b else '0' for b in diff.flatten())
        
        # Convert binary string to hex
        hash_hex = hex(int(hash_bits, 2))[2:].rjust(len(hash_bits)//4, '0')
        
        logger.debug("Generated dHash", image_path=image_path, hash_size=hash_size)
        return hash_hex
        
    except Exception as e:
        logger.error("Failed to generate dHash", image_path=image_path, error=str(e))
        raise

def phash(image_path: str, hash_size: int = 8) -> str:
    """
    Generate perceptual hash (pHash) using DCT.
    Very robust for detecting duplicates with various modifications.
    """
    try:
        # Load and convert image  
        image = Image.open(image_path).convert('L')
        
        # Resize to hash_size * 4 for better DCT
        image = image.resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(image, dtype=np.float32)
        
        # Apply DCT (Discrete Cosine Transform)
        dct = cv2.dct(pixels)
        
        # Extract top-left hash_size x hash_size corner (low frequencies)
        dct_low = dct[:hash_size, :hash_size] 
        
        # Calculate median
        median = np.median(dct_low)
        
        # Generate hash based on median comparison  
        hash_bits = ''.join('1' if pixel > median else '0' for pixel in dct_low.flatten())
        
        # Convert to hex
        hash_hex = hex(int(hash_bits, 2))[2:].rjust(len(hash_bits)//4, '0')
        
        logger.debug("Generated pHash", image_path=image_path, hash_size=hash_size)
        return hash_hex
        
    except Exception as e:
        logger.error("Failed to generate pHash", image_path=image_path, error=str(e))
        raise

def ahash(image_path: str, hash_size: int = 8) -> str:
    """
    Generate average hash (aHash).
    Simple and fast, good for basic duplicate detection.
    """
    try:
        # Load and convert image
        image = Image.open(image_path).convert('L')
        
        # Resize to hash_size x hash_size
        image = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(image)
        
        # Calculate average
        avg = np.mean(pixels)
        
        # Generate hash based on average comparison
        hash_bits = ''.join('1' if pixel > avg else '0' for pixel in pixels.flatten())
        
        # Convert to hex
        hash_hex = hex(int(hash_bits, 2))[2:].rjust(len(hash_bits)//4, '0')
        
        logger.debug("Generated aHash", image_path=image_path, hash_size=hash_size)
        return hash_hex
        
    except Exception as e:
        logger.error("Failed to generate aHash", image_path=image_path, error=str(e))
        raise

def generate_image_hashes(image_path: str) -> dict:
    """
    Generate multiple perceptual hashes for robust duplicate detection.
    
    Returns:
        Dictionary containing different hash types
    """
    try:
        hashes = {
            'dhash': dhash(image_path),
            'phash': phash(image_path), 
            'ahash': ahash(image_path),
            'dhash_16': dhash(image_path, 16),  # Higher resolution
            'phash_16': phash(image_path, 16)   # Higher resolution
        }
        
        logger.info("Generated image hashes", 
                   image_path=image_path, 
                   hash_types=list(hashes.keys()))
        return hashes
        
    except Exception as e:
        logger.error("Failed to generate image hashes", 
                    image_path=image_path, error=str(e))
        raise

def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hash strings."""
    if len(hash1) != len(hash2):
        return float('inf')  # Different lengths = not comparable
    
    # Convert hex to binary and compare
    try:
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
    except ValueError:
        return float('inf')

def calculate_hash_similarity(hashes1: dict, hashes2: dict) -> Tuple[float, str]:
    """
    Calculate similarity between two sets of image hashes.
    
    Returns:
        Tuple of (similarity_score, best_match_type)
    """
    similarities = {}
    
    for hash_type in hashes1:
        if hash_type in hashes2:
            distance = hamming_distance(hashes1[hash_type], hashes2[hash_type])
            
            # Calculate similarity (lower distance = higher similarity)
            # For 64-bit hashes (8x8), max distance is 64
            # For 256-bit hashes (16x16), max distance is 256
            max_distance = len(hashes1[hash_type]) * 4  # 4 bits per hex char
            similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 0.0
            similarities[hash_type] = similarity
    
    if not similarities:
        return 0.0, "none"
    
    # Return the best similarity score and its type
    best_type = max(similarities, key=similarities.get)
    best_score = similarities[best_type]
    
    logger.debug("Hash similarity calculated", 
                similarities=similarities, 
                best_type=best_type, 
                best_score=best_score)
    
    return best_score, best_type

def is_duplicate_by_hash(image_path1: str, image_path2: str, threshold: float = 0.95) -> Tuple[bool, float, str]:
    """
    Check if two images are duplicates using perceptual hashing.
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image  
        threshold: Similarity threshold for duplicate detection
        
    Returns:
        Tuple of (is_duplicate, similarity_score, match_type)
    """
    try:
        hashes1 = generate_image_hashes(image_path1)
        hashes2 = generate_image_hashes(image_path2)
        
        similarity, match_type = calculate_hash_similarity(hashes1, hashes2)
        is_duplicate = similarity >= threshold
        
        logger.info("Perceptual hash comparison completed",
                   image1=image_path1,
                   image2=image_path2, 
                   similarity=similarity,
                   match_type=match_type,
                   is_duplicate=is_duplicate,
                   threshold=threshold)
        
        return is_duplicate, similarity, match_type
        
    except Exception as e:
        logger.error("Failed to compare image hashes", 
                    image1=image_path1, 
                    image2=image_path2, 
                    error=str(e))
        return False, 0.0, "error" 