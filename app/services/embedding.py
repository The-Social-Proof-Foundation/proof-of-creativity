import torch
import numpy as np
import structlog
from PIL import Image
from typing import List, Optional, Union
from functools import lru_cache
from transformers import CLIPProcessor, CLIPModel
import os

logger = structlog.get_logger()

# Global model and processor instances for reuse
_clip_model = None
_clip_processor = None
_device = None

def get_device() -> torch.device:
    """Get the best available device for inference."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            logger.info("Using CUDA GPU for embeddings")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available():
            _device = torch.device("mps")
            logger.info("Using Apple Silicon MPS for embeddings")
        else:
            _device = torch.device("cpu")
            logger.info("Using CPU for embeddings")
    return _device

def load_clip_model(model_name: str = None) -> tuple:
    """Load CLIP model and processor with caching."""
    global _clip_model, _clip_processor
    
    if _clip_model is None or _clip_processor is None:
        try:
            model_name = model_name or os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
            device = get_device()
            
            logger.info("Loading CLIP model", model_name=model_name, device=str(device))
            
            _clip_processor = CLIPProcessor.from_pretrained(model_name)
            _clip_model = CLIPModel.from_pretrained(model_name)
            _clip_model.to(device)
            _clip_model.eval()  # Set to evaluation mode
            
            logger.info("CLIP model loaded successfully", 
                       model_name=model_name, 
                       device=str(device),
                       parameters=sum(p.numel() for p in _clip_model.parameters()))
                       
        except Exception as e:
            logger.error("Failed to load CLIP model", model_name=model_name, error=str(e))
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
    
    return _clip_model, _clip_processor

def preprocess_image(image_path: str, max_size: tuple = (512, 512)) -> Image.Image:
    """Preprocess image for embedding generation."""
    try:
        # Load and validate image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            logger.debug("Converting image to RGB", original_mode=image.mode)
            image = image.convert('RGB')
        
        # Resize if too large (for memory efficiency)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            logger.debug("Resizing large image", 
                        original_size=image.size, max_size=max_size)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        logger.error("Failed to preprocess image", image_path=image_path, error=str(e))
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def image_embedding(path: str, model_name: str = None, normalize: bool = True) -> List[float]:
    """
    Generate CLIP embedding for an image.
    
    Args:
        path: Path to the image file
        model_name: Optional model name override
        normalize: Whether to normalize the embedding vector
        
    Returns:
        List of floats representing the image embedding
    """
    try:
        logger.debug("Generating image embedding", image_path=path)
        
        # Load model and processor
        model, processor = load_clip_model(model_name)
        device = get_device()
        
        # Preprocess image
        image = preprocess_image(path)
        
        # Process image for CLIP
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
            # Normalize if requested
            if normalize:
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            
            # Convert to list
            embedding = image_features.cpu().numpy()[0].tolist()
        
        logger.debug("Image embedding generated successfully", 
                    image_path=path, embedding_dim=len(embedding))
        return embedding
        
    except Exception as e:
        logger.error("Failed to generate image embedding", 
                    image_path=path, error=str(e))
        raise RuntimeError(f"Failed to generate image embedding: {str(e)}")

def batch_image_embeddings(image_paths: List[str], 
                          model_name: str = None, 
                          batch_size: int = 8,
                          normalize: bool = True) -> List[List[float]]:
    """
    Generate embeddings for multiple images efficiently.
    
    Args:
        image_paths: List of paths to image files
        model_name: Optional model name override
        batch_size: Number of images to process at once
        normalize: Whether to normalize embedding vectors
        
    Returns:
        List of embedding vectors
    """
    try:
        logger.info("Generating batch image embeddings", 
                   count=len(image_paths), batch_size=batch_size)
        
        # Load model and processor
        model, processor = load_clip_model(model_name)
        device = get_device()
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = preprocess_image(path)
                    batch_images.append(image)
                except Exception as e:
                    logger.warning("Skipping failed image in batch", 
                                 image_path=path, error=str(e))
                    # Add None as placeholder
                    batch_images.append(None)
            
            # Filter out failed images
            valid_images = [img for img in batch_images if img is not None]
            
            if not valid_images:
                logger.warning("No valid images in batch", batch_start=i)
                # Add empty embeddings for failed images
                embeddings.extend([[] for _ in batch_images])
                continue
            
            # Process valid images
            inputs = processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                batch_features = model.get_image_features(**inputs)
                
                if normalize:
                    batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
                
                batch_embeddings = batch_features.cpu().numpy().tolist()
            
            # Map embeddings back to original order (accounting for failed images)
            j = 0
            for img in batch_images:
                if img is not None:
                    embeddings.append(batch_embeddings[j])
                    j += 1
                else:
                    embeddings.append([])  # Empty embedding for failed image
        
        logger.info("Batch image embeddings completed", 
                   total_count=len(image_paths), 
                   successful_count=len([e for e in embeddings if e]))
        
        return embeddings
        
    except Exception as e:
        logger.error("Failed to generate batch image embeddings", 
                    count=len(image_paths), error=str(e))
        raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")

def text_embedding(text: str, model_name: str = None, normalize: bool = True) -> List[float]:
    """
    Generate CLIP embedding for text (useful for text-based similarity search).
    
    Args:
        text: Text to embed
        model_name: Optional model name override
        normalize: Whether to normalize the embedding vector
        
    Returns:
        List of floats representing the text embedding
    """
    try:
        logger.debug("Generating text embedding", text_length=len(text))
        
        # Load model and processor
        model, processor = load_clip_model(model_name)
        device = get_device()
        
        # Process text for CLIP
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            
            if normalize:
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            
            embedding = text_features.cpu().numpy()[0].tolist()
        
        logger.debug("Text embedding generated successfully", 
                    text_length=len(text), embedding_dim=len(embedding))
        return embedding
        
    except Exception as e:
        logger.error("Failed to generate text embedding", error=str(e))
        raise RuntimeError(f"Failed to generate text embedding: {str(e)}")

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    try:
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        logger.error("Failed to calculate cosine similarity", error=str(e))
        return 0.0

def get_embedding_info() -> dict:
    """Get information about the loaded embedding model."""
    try:
        model, processor = load_clip_model()
        device = get_device()
        
        return {
            "model_name": processor.feature_extractor.model_name if hasattr(processor, 'feature_extractor') else "unknown",
            "embedding_dimension": 512,  # CLIP ViT-B/32 default
            "device": str(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "memory_usage_mb": torch.cuda.memory_allocated() / 1024 / 1024 if device.type == "cuda" else "N/A"
        }
    except Exception as e:
        logger.error("Failed to get embedding info", error=str(e))
        return {"error": str(e)}

# Cache for frequently accessed embeddings (development/testing)
@lru_cache(maxsize=100)
def cached_image_embedding(path: str, model_name: str = None) -> tuple:
    """Cached version of image_embedding for development use."""
    embedding = image_embedding(path, model_name)
    return tuple(embedding)  # Convert to tuple for hashing

def clear_embedding_cache():
    """Clear the embedding cache."""
    cached_image_embedding.cache_clear()
    logger.info("Embedding cache cleared")

def warmup_model(model_name: str = None):
    """Warm up the model with a dummy inference to reduce first-call latency."""
    try:
        logger.info("Warming up embedding model")
        
        # Create a small dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        dummy_path = "/tmp/dummy_warmup.jpg"
        dummy_image.save(dummy_path)
        
        try:
            # Run dummy inference
            _ = image_embedding(dummy_path, model_name)
            logger.info("Model warmup completed successfully")
        finally:
            # Clean up dummy file
            try:
                os.unlink(dummy_path)
            except:
                pass
                
    except Exception as e:
        logger.warning("Model warmup failed", error=str(e))
