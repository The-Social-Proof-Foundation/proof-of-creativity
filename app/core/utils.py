import uuid
import hashlib
import os
import tempfile
import structlog
from pathlib import Path
from typing import Optional

logger = structlog.get_logger()

def save_temp_upload(upload_file) -> str:
    """Save uploaded file to temporary location and return path."""
    try:
        suffix = Path(upload_file.filename).suffix if upload_file.filename else ""
        
        # Create temporary file with proper suffix
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        
        try:
            # Write file content
            with os.fdopen(temp_fd, 'wb') as f:
                # Read file content
                content = upload_file.file.read()
                f.write(content)
            
            # Reset file pointer for potential re-reading
            upload_file.file.seek(0)
            
            logger.info("Saved temporary upload", 
                       filename=upload_file.filename, temp_path=temp_path, size=len(content))
            return temp_path
            
        except Exception as e:
            # Clean up file descriptor if writing failed
            try:
                os.close(temp_fd)
            except:
                pass
            raise e
            
    except Exception as e:
        logger.error("Failed to save temporary upload", 
                    filename=getattr(upload_file, 'filename', 'unknown'), error=str(e))
        raise

def new_media_id() -> str:
    """Generate a new unique media ID."""
    return str(uuid.uuid4())

def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate hash of file content for deduplication."""
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        file_hash = hash_obj.hexdigest()
        logger.debug("Calculated file hash", file_path=file_path, hash=file_hash, algorithm=algorithm)
        return file_hash
        
    except Exception as e:
        logger.error("Failed to calculate file hash", file_path=file_path, error=str(e))
        raise

def cleanup_temp_file(file_path: str) -> bool:
    """Clean up temporary file safely."""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug("Cleaned up temporary file", file_path=file_path)
            return True
        return False
    except Exception as e:
        logger.warning("Failed to cleanup temporary file", file_path=file_path, error=str(e))
        return False

def ensure_dir_exists(dir_path: str) -> str:
    """Ensure directory exists, create if it doesn't."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return dir_path
    except Exception as e:
        logger.error("Failed to create directory", dir_path=dir_path, error=str(e))
        raise

def get_file_mime_type(file_path: str) -> Optional[str]:
    """Get MIME type of file using python-magic if available, fallback to mimetypes."""
    import mimetypes
    
    try:
        # Try python-magic first for more accurate detection
        try:
            import magic
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except ImportError:
            # Fallback to standard mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type
    except Exception as e:
        logger.warning("Failed to determine MIME type", file_path=file_path, error=str(e))
        return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def is_media_file(filename: str) -> tuple[bool, Optional[str]]:
    """Check if file is a supported media file and return media type."""
    if not filename:
        return False, None
    
    # Convert to lowercase for comparison
    filename_lower = filename.lower()
    
    # Image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
    if any(filename_lower.endswith(ext) for ext in image_extensions):
        return True, 'image'
    
    # Audio extensions
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    if any(filename_lower.endswith(ext) for ext in audio_extensions):
        return True, 'audio'
    
    # Video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv'}
    if any(filename_lower.endswith(ext) for ext in video_extensions):
        return True, 'video'
    
    return False, None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    if not filename:
        return "unnamed_file"
    
    # Remove or replace problematic characters
    import re
    
    # Keep only alphanumeric, dots, dashes, underscores
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_{2,}', '_', sanitized)
    
    # Ensure it doesn't start with a dot (hidden file)
    if sanitized.startswith('.'):
        sanitized = 'file_' + sanitized
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized

def create_media_storage_path(media_id: str, filename: str, media_type: str) -> str:
    """Create a structured storage path for media files."""
    sanitized_filename = sanitize_filename(filename)
    
    # Create path structure: media_type/year/month/media_id_filename
    from datetime import datetime
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    
    storage_path = f"{media_type}/{year}/{month}/{media_id}_{sanitized_filename}"
    return storage_path

def validate_media_dimensions(file_path: str, media_type: str) -> dict:
    """Validate and get media dimensions/properties."""
    try:
        if media_type == "image":
            from PIL import Image
            with Image.open(file_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode
                }
        elif media_type == "video":
            import cv2
            cap = cv2.VideoCapture(file_path)
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                return {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration_seconds": duration
                }
            finally:
                cap.release()
        elif media_type == "audio":
            import librosa
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            return {
                "sample_rate": sr,
                "duration_seconds": duration,
                "channels": 1 if len(y.shape) == 1 else y.shape[0]
            }
    except Exception as e:
        logger.warning("Failed to validate media dimensions", 
                      file_path=file_path, media_type=media_type, error=str(e))
        return {}

def batch_cleanup_temp_files(file_paths: list[str]) -> int:
    """Clean up multiple temporary files, return count of successfully cleaned files."""
    cleaned_count = 0
    for file_path in file_paths:
        if cleanup_temp_file(file_path):
            cleaned_count += 1
    return cleaned_count
