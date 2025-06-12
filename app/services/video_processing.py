import tempfile
import subprocess
import os
import structlog
from pathlib import Path
from typing import List, Optional, Tuple
import shutil

from app.services import embedding
from app.services import fingerprint
from app.core.utils import cleanup_temp_file, batch_cleanup_temp_files

logger = structlog.get_logger()

DEFAULT_KEYFRAME_RATE = 1  # frames per second
DEFAULT_AUDIO_SAMPLE_RATE = 8000
MAX_FRAMES_TO_PROCESS = 300  # Limit for very long videos
MAX_VIDEO_DURATION = 600  # 10 minutes max


def validate_video_file(video_path: str) -> dict:
    """Validate video file and get basic information."""
    try:
        # Use ffprobe to get video information
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = None
        audio_stream = None
        
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream
        
        if not video_stream:
            raise ValueError("No video stream found in file")
        
        duration = float(info.get("format", {}).get("duration", 0))
        
        if duration > MAX_VIDEO_DURATION:
            logger.warning("Video exceeds maximum duration", 
                         duration=duration, max_duration=MAX_VIDEO_DURATION)
        
        validation_info = {
            "has_video": video_stream is not None,
            "has_audio": audio_stream is not None,
            "duration": duration,
            "width": int(video_stream.get("width", 0)) if video_stream else 0,
            "height": int(video_stream.get("height", 0)) if video_stream else 0,
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        }
        
        logger.info("Video validation completed", **validation_info)
        return validation_info
        
    except subprocess.CalledProcessError as e:
        logger.error("FFprobe failed", video_path=video_path, error=e.stderr)
        raise ValueError(f"Invalid video file: {e.stderr}")
    except Exception as e:
        logger.error("Video validation failed", video_path=video_path, error=str(e))
        raise ValueError(f"Video validation failed: {str(e)}")


def extract_keyframes(
    video_path: str, 
    rate: int = DEFAULT_KEYFRAME_RATE,
    max_frames: int = MAX_FRAMES_TO_PROCESS
) -> List[str]:
    """Extract keyframes from video at specified rate."""
    temp_dir = None
    try:
        logger.info("Extracting keyframes", 
                   video_path=video_path, rate=rate, max_frames=max_frames)
        
        # Validate video first
        video_info = validate_video_file(video_path)
        
        # Calculate expected number of frames
        duration = video_info.get("duration", 0)
        expected_frames = int(duration * rate)
        
        if expected_frames > max_frames:
            # Adjust rate to stay within limits
            adjusted_rate = max_frames / duration
            logger.info("Adjusting keyframe rate to stay within limits", 
                       original_rate=rate, adjusted_rate=adjusted_rate, 
                       expected_frames=expected_frames, max_frames=max_frames)
            rate = adjusted_rate
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix="keyframes_")
        frame_pattern = Path(temp_dir) / "frame_%06d.jpg"
        
        # Extract keyframes using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={rate}",
            "-q:v", "2",  # High quality JPEG
            "-an",  # No audio
            str(frame_pattern),
            "-y",  # Overwrite output files
            "-loglevel", "error"
        ]
        
        logger.debug("Running ffmpeg command", cmd=" ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find extracted frames
        frame_files = sorted(Path(temp_dir).glob("frame_*.jpg"))
        frame_paths = [str(f) for f in frame_files]
        
        if not frame_paths:
            raise RuntimeError("No keyframes were extracted from video")
        
        logger.info("Keyframes extracted successfully", 
                   video_path=video_path, frame_count=len(frame_paths))
        
        return frame_paths
        
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg keyframe extraction failed", 
                    video_path=video_path, error=e.stderr)
        raise RuntimeError(f"Keyframe extraction failed: {e.stderr}")
    except Exception as e:
        logger.error("Keyframe extraction failed", 
                    video_path=video_path, error=str(e))
        # Clean up on failure
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Keyframe extraction failed: {str(e)}")


def extract_audio(
    video_path: str, 
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> Optional[str]:
    """Extract audio track from video."""
    try:
        logger.info("Extracting audio", 
                   video_path=video_path, sample_rate=sample_rate)
        
        # Validate video and check for audio
        video_info = validate_video_file(video_path)
        
        if not video_info.get("has_audio"):
            logger.warning("Video has no audio track", video_path=video_path)
            return None
        
        # Create temporary file for audio
        temp_file = tempfile.mktemp(suffix=".wav")
        
        # Extract audio using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-map", "a:0",  # First audio stream
            "-ac", "1",     # Mono audio
            "-ar", str(sample_rate),  # Sample rate
            "-acodec", "pcm_s16le",   # PCM 16-bit
            temp_file,
            "-y",  # Overwrite output file
            "-loglevel", "error"
        ]
        
        logger.debug("Running ffmpeg audio extraction", cmd=" ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Verify audio file was created
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            raise RuntimeError("Audio extraction produced empty file")
        
        logger.info("Audio extracted successfully", 
                   video_path=video_path, audio_path=temp_file)
        
        return temp_file
        
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg audio extraction failed", 
                    video_path=video_path, error=e.stderr)
        return None  # Audio extraction failure is not fatal
    except Exception as e:
        logger.error("Audio extraction failed", 
                    video_path=video_path, error=str(e))
        return None  # Audio extraction failure is not fatal


def process_video(
    video_path: str, 
    keyframe_rate: int = DEFAULT_KEYFRAME_RATE,
    max_frames: int = MAX_FRAMES_TO_PROCESS
) -> Tuple[List[List[float]], Optional[str]]:
    """
    Process video to extract keyframe embeddings and audio fingerprint.
    
    Args:
        video_path: Path to video file
        keyframe_rate: Frames per second to extract
        max_frames: Maximum number of frames to process
        
    Returns:
        Tuple of (frame_embeddings, audio_fingerprint_hash)
    """
    frame_paths = []
    audio_path = None
    
    try:
        logger.info("Processing video", 
                   video_path=video_path, keyframe_rate=keyframe_rate)
        
        # Validate video file
        video_info = validate_video_file(video_path)
        logger.info("Video info", **video_info)
        
        # Extract keyframes
        frame_paths = extract_keyframes(video_path, keyframe_rate, max_frames)
        
        # Generate embeddings for keyframes
        if frame_paths:
            logger.info("Generating embeddings for keyframes", frame_count=len(frame_paths))
            # Use batch processing for efficiency
            frame_embeddings = embedding.batch_image_embeddings(
                frame_paths, 
                batch_size=8  # Process 8 frames at a time
            )
            
            # Filter out empty embeddings (failed frames)
            valid_embeddings = [emb for emb in frame_embeddings if emb]
            
            if len(valid_embeddings) != len(frame_embeddings):
                logger.warning("Some keyframes failed to generate embeddings", 
                             total_frames=len(frame_embeddings),
                             valid_embeddings=len(valid_embeddings))
        else:
            logger.warning("No keyframes extracted", video_path=video_path)
            valid_embeddings = []
        
        # Extract and fingerprint audio
        audio_fingerprint_hash = None
        if video_info.get("has_audio"):
            audio_path = extract_audio(video_path)
            if audio_path:
                try:
                    audio_fingerprint_hash = fingerprint.fingerprint_audio(audio_path)
                    logger.info("Audio fingerprint generated", 
                               fingerprint_hash=audio_fingerprint_hash)
                except Exception as e:
                    logger.error("Audio fingerprinting failed", 
                               audio_path=audio_path, error=str(e))
                    audio_fingerprint_hash = None
        
        logger.info("Video processing completed successfully", 
                   video_path=video_path,
                   frame_embeddings_count=len(valid_embeddings),
                   has_audio_fingerprint=audio_fingerprint_hash is not None)
        
        return valid_embeddings, audio_fingerprint_hash
        
    except Exception as e:
        logger.error("Video processing failed", 
                    video_path=video_path, error=str(e))
        raise RuntimeError(f"Video processing failed: {str(e)}")
        
    finally:
        # Clean up temporary files
        cleanup_temp_files = []
        
        # Clean up frame files and directory
        if frame_paths:
            cleanup_temp_files.extend(frame_paths)
            # Also remove the temporary directory
            for frame_path in frame_paths:
                temp_dir = os.path.dirname(frame_path)
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        logger.debug("Cleaned up frames directory", temp_dir=temp_dir)
                    except Exception as e:
                        logger.warning("Failed to cleanup frames directory", 
                                     temp_dir=temp_dir, error=str(e))
                break  # Only need to clean up once
        
        # Clean up audio file
        if audio_path:
            cleanup_temp_files.append(audio_path)
        
        # Batch cleanup
        if cleanup_temp_files:
            cleaned_count = batch_cleanup_temp_files(cleanup_temp_files)
            logger.debug("Cleanup completed", 
                        total_files=len(cleanup_temp_files), 
                        cleaned_files=cleaned_count)


def get_video_thumbnail(video_path: str, timestamp: float = 5.0) -> Optional[str]:
    """Extract a single thumbnail from video at specified timestamp."""
    try:
        logger.debug("Extracting video thumbnail", 
                    video_path=video_path, timestamp=timestamp)
        
        # Validate video
        video_info = validate_video_file(video_path)
        duration = video_info.get("duration", 0)
        
        # Adjust timestamp if it exceeds video duration
        if timestamp >= duration:
            timestamp = duration / 2  # Use middle of video
        
        # Create temporary file for thumbnail
        temp_file = tempfile.mktemp(suffix=".jpg")
        
        # Extract thumbnail using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(timestamp),  # Seek to timestamp
            "-vframes", "1",        # Extract 1 frame
            "-q:v", "2",           # High quality
            temp_file,
            "-y",  # Overwrite
            "-loglevel", "error"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            logger.debug("Video thumbnail extracted successfully", 
                        video_path=video_path, thumbnail_path=temp_file)
            return temp_file
        else:
            logger.warning("Thumbnail extraction produced empty file", 
                          video_path=video_path)
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg thumbnail extraction failed", 
                    video_path=video_path, error=e.stderr)
        return None
    except Exception as e:
        logger.error("Thumbnail extraction failed", 
                    video_path=video_path, error=str(e))
        return None


def check_ffmpeg_installation() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        logger.info("FFmpeg is available", version_info=result.stdout.split('\n')[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not accessible")
        return False


def get_supported_video_formats() -> List[str]:
    """Get list of video formats supported by the current FFmpeg installation."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-formats"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse output to find video formats
        formats = []
        in_video_section = False
        
        for line in result.stdout.split('\n'):
            if 'File formats:' in line:
                in_video_section = True
                continue
            
            if in_video_section and line.strip():
                # Format line looks like: " DE mp4             MP4 (MPEG-4 Part 14)"
                if line.startswith(' ') and 'E' in line[:10]:  # Can encode
                    parts = line.split()
                    if len(parts) >= 2:
                        format_name = parts[1]
                        formats.append(format_name)
        
        return formats[:20]  # Return first 20 formats
        
    except Exception as e:
        logger.warning("Could not get supported video formats", error=str(e))
        return ["mp4", "avi", "mov", "webm"]  # Default common formats
