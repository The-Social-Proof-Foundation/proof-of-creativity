import numpy as np
import librosa
import hashlib
import structlog
from typing import List, Tuple, Dict, Optional, Set
from scipy import signal
from collections import defaultdict
import pickle
import os

logger = structlog.get_logger()

# Fingerprinting configuration
DEFAULT_SAMPLE_RATE = 8000
WINDOW_SIZE = 2048
HOP_LENGTH = 512
N_MELS = 64
PEAK_THRESHOLD = 0.1
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FAN_VALUE = 5
FINGERPRINT_REDUCTION = 20

class AudioFingerprinter:
    """Enhanced audio fingerprinting using spectral peak pair hashing."""
    
    def __init__(self, 
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 window_size: int = WINDOW_SIZE,
                 hop_length: int = HOP_LENGTH):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        
        logger.info("AudioFingerprinter initialized", 
                   sample_rate=sample_rate, 
                   window_size=window_size, 
                   hop_length=hop_length)
    
    def fingerprint_audio(self, file_path: str, max_duration: float = 60.0) -> Tuple[str, List[Tuple]]:
        """
        Generate fingerprint for audio file.
        
        Args:
            file_path: Path to audio file
            max_duration: Maximum duration to process (seconds)
            
        Returns:
            Tuple of (primary_hash, detailed_fingerprints)
        """
        try:
            logger.info("Starting audio fingerprinting", file_path=file_path)
            
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Limit duration for performance
            if len(y) / sr > max_duration:
                max_samples = int(max_duration * sr)
                y = y[:max_samples]
                logger.info("Truncated audio for processing", 
                           original_duration=len(y)/sr, max_duration=max_duration)
            
            # Generate spectrogram
            spectrogram = self._generate_spectrogram(y)
            
            # Find spectral peaks
            peaks = self._find_spectral_peaks(spectrogram)
            
            # Generate constellation map
            constellation = self._create_constellation_map(peaks)
            
            # Generate hash pairs
            hashes = self._generate_hash_pairs(constellation)
            
            # Create primary hash for quick lookup
            primary_hash = self._create_primary_hash(hashes)
            
            logger.info("Audio fingerprinting completed", 
                       file_path=file_path,
                       duration=len(y)/sr,
                       peaks_found=len(peaks),
                       hashes_generated=len(hashes),
                       primary_hash=primary_hash)
            
            return primary_hash, hashes
            
        except Exception as e:
            logger.error("Audio fingerprinting failed", file_path=file_path, error=str(e))
            raise
    
    def _generate_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Generate mel-scaled spectrogram."""
        try:
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_fft=self.window_size,
                hop_length=self.hop_length,
                n_mels=N_MELS,
                power=2.0
            )
            
            # Convert to decibels
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            logger.debug("Spectrogram generated", 
                        shape=mel_spec_db.shape,
                        freq_bins=mel_spec_db.shape[0],
                        time_bins=mel_spec_db.shape[1])
            
            return mel_spec_db
            
        except Exception as e:
            logger.error("Spectrogram generation failed", error=str(e))
            raise
    
    def _find_spectral_peaks(self, spectrogram: np.ndarray) -> List[Tuple[int, int]]:
        """Find spectral peaks in the spectrogram."""
        try:
            peaks = []
            
            # Apply peak detection across frequency and time
            for t in range(spectrogram.shape[1]):
                for f in range(spectrogram.shape[0]):
                    # Check if current point is a local maximum
                    if self._is_local_peak(spectrogram, f, t):
                        peaks.append((f, t))
            
            # Filter peaks by threshold
            threshold = np.percentile(spectrogram, 95)  # Top 5% of values
            filtered_peaks = []
            
            for f, t in peaks:
                if spectrogram[f, t] > threshold:
                    filtered_peaks.append((f, t))
            
            # Reduce peaks to manageable number
            if len(filtered_peaks) > 1000:
                # Sort by magnitude and take top peaks
                peak_magnitudes = [(spectrogram[f, t], f, t) for f, t in filtered_peaks]
                peak_magnitudes.sort(reverse=True)
                filtered_peaks = [(f, t) for _, f, t in peak_magnitudes[:1000]]
            
            logger.debug("Spectral peaks found", 
                        total_candidates=len(peaks),
                        filtered_peaks=len(filtered_peaks),
                        threshold=threshold)
            
            return filtered_peaks
            
        except Exception as e:
            logger.error("Peak detection failed", error=str(e))
            raise
    
    def _is_local_peak(self, spectrogram: np.ndarray, f: int, t: int, 
                      neighborhood_size: int = 3) -> bool:
        """Check if a point is a local peak in its neighborhood."""
        try:
            # Define neighborhood bounds
            f_min = max(0, f - neighborhood_size)
            f_max = min(spectrogram.shape[0], f + neighborhood_size + 1)
            t_min = max(0, t - neighborhood_size)
            t_max = min(spectrogram.shape[1], t + neighborhood_size + 1)
            
            # Get neighborhood
            neighborhood = spectrogram[f_min:f_max, t_min:t_max]
            
            # Check if center point is maximum in neighborhood
            center_value = spectrogram[f, t]
            return center_value == np.max(neighborhood) and center_value > PEAK_THRESHOLD
            
        except Exception as e:
            logger.debug("Local peak check failed", f=f, t=t, error=str(e))
            return False
    
    def _create_constellation_map(self, peaks: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """Create constellation map with timing information."""
        try:
            constellation = []
            
            for f, t in peaks:
                # Convert time bin to actual time
                time_seconds = t * self.hop_length / self.sample_rate
                constellation.append((f, t, time_seconds))
            
            # Sort by time
            constellation.sort(key=lambda x: x[1])  # Sort by time bin
            
            logger.debug("Constellation map created", points=len(constellation))
            
            return constellation
            
        except Exception as e:
            logger.error("Constellation map creation failed", error=str(e))
            raise
    
    def _generate_hash_pairs(self, constellation: List[Tuple[int, int, float]]) -> List[Tuple]:
        """Generate hash pairs from constellation points (Shazam-style)."""
        try:
            hashes = []
            
            for i, (f1, t1, time1) in enumerate(constellation):
                # Look ahead for target peaks within time window
                for j in range(i + 1, min(i + FAN_VALUE + 1, len(constellation))):
                    f2, t2, time2 = constellation[j]
                    
                    # Calculate time delta
                    time_delta = t2 - t1
                    
                    # Skip if time delta is outside acceptable range
                    if time_delta < MIN_HASH_TIME_DELTA or time_delta > MAX_HASH_TIME_DELTA:
                        continue
                    
                    # Create hash from frequency pair and time delta
                    hash_value = self._create_hash(f1, f2, time_delta)
                    
                    # Store hash with anchor time for offset calculation
                    hashes.append((hash_value, t1, f1, f2, time_delta))
            
            logger.debug("Hash pairs generated", count=len(hashes))
            
            return hashes
            
        except Exception as e:
            logger.error("Hash pair generation failed", error=str(e))
            raise
    
    def _create_hash(self, f1: int, f2: int, time_delta: int) -> str:
        """Create hash from frequency pair and time delta."""
        try:
            # Create hash string from frequency pair and time delta
            hash_input = f"{f1}|{f2}|{time_delta}"
            
            # Use SHA-1 for consistent hashing (like Shazam)
            hash_object = hashlib.sha1(hash_input.encode())
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.error("Hash creation failed", f1=f1, f2=f2, time_delta=time_delta, error=str(e))
            raise
    
    def _create_primary_hash(self, hashes: List[Tuple]) -> str:
        """Create primary hash for quick database lookup."""
        try:
            if not hashes:
                return hashlib.sha1(b"empty").hexdigest()
            
            # Take first few hashes and create composite hash
            primary_hashes = [h[0] for h in hashes[:FINGERPRINT_REDUCTION]]
            combined_hash = "|".join(sorted(primary_hashes))
            
            return hashlib.sha1(combined_hash.encode()).hexdigest()
            
        except Exception as e:
            logger.error("Primary hash creation failed", error=str(e))
            raise
    
    def match_fingerprints(self, 
                          query_hashes: List[Tuple], 
                          target_hashes: List[Tuple],
                          min_matching_hashes: int = 5) -> Dict:
        """
        Match two sets of fingerprint hashes.
        
        Args:
            query_hashes: Hashes from query audio
            target_hashes: Hashes from target audio
            min_matching_hashes: Minimum number of matching hashes for valid match
            
        Returns:
            Dictionary with match information
        """
        try:
            # Create hash lookup for target
            target_hash_dict = defaultdict(list)
            for hash_val, anchor_time, f1, f2, time_delta in target_hashes:
                target_hash_dict[hash_val].append(anchor_time)
            
            # Find matching hashes and calculate time offsets
            matches = defaultdict(int)
            matching_hashes = 0
            
            for query_hash, query_anchor, f1, f2, time_delta in query_hashes:
                if query_hash in target_hash_dict:
                    matching_hashes += 1
                    
                    # Calculate time offset for each match
                    for target_anchor in target_hash_dict[query_hash]:
                        offset = target_anchor - query_anchor
                        matches[offset] += 1
            
            if matching_hashes < min_matching_hashes:
                return {
                    "match": False,
                    "confidence": 0.0,
                    "matching_hashes": matching_hashes,
                    "total_query_hashes": len(query_hashes),
                    "offset": None
                }
            
            # Find the most likely offset
            best_offset = max(matches.keys(), key=matches.get) if matches else 0
            best_match_count = matches[best_offset] if matches else 0
            
            # Calculate confidence
            confidence = min(1.0, best_match_count / max(1, len(query_hashes) * 0.1))
            
            return {
                "match": True,
                "confidence": confidence,
                "matching_hashes": matching_hashes,
                "total_query_hashes": len(query_hashes),
                "offset": best_offset,
                "offset_matches": best_match_count,
                "all_offsets": dict(matches)
            }
            
        except Exception as e:
            logger.error("Fingerprint matching failed", error=str(e))
            return {"match": False, "error": str(e)}

# Global fingerprinter instance
_fingerprinter = AudioFingerprinter()

def fingerprint_audio(file_path: str) -> str:
    """
    Legacy function for backward compatibility.
    Generate simple hash for audio file.
    """
    try:
        logger.info("Generating simple audio fingerprint", file_path=file_path)
        
        # Generate detailed fingerprint
        primary_hash, detailed_hashes = _fingerprinter.fingerprint_audio(file_path)
        
        return primary_hash
        
    except Exception as e:
        logger.error("Simple fingerprinting failed", file_path=file_path, error=str(e))
        # Fallback to basic hash
        try:
            y, sr = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
            S = np.abs(librosa.stft(y))
            S = librosa.feature.melspectrogram(S=S, sr=sr)
            fp = hashlib.sha1(S.tobytes()).hexdigest()
            return fp
        except Exception as fallback_error:
            logger.error("Fallback fingerprinting failed", error=str(fallback_error))
            raise

def detailed_fingerprint_audio(file_path: str) -> Tuple[str, List[Tuple]]:
    """
    Generate detailed audio fingerprint with hash pairs.
    
    Returns:
        Tuple of (primary_hash, detailed_fingerprint_data)
    """
    return _fingerprinter.fingerprint_audio(file_path)

def match_audio_fingerprints(query_file: str, target_file: str) -> Dict:
    """
    Match two audio files using detailed fingerprinting.
    
    Args:
        query_file: Path to query audio file
        target_file: Path to target audio file
        
    Returns:
        Dictionary with match results
    """
    try:
        logger.info("Matching audio fingerprints", 
                   query_file=query_file, target_file=target_file)
        
        # Generate fingerprints for both files
        _, query_hashes = _fingerprinter.fingerprint_audio(query_file)
        _, target_hashes = _fingerprinter.fingerprint_audio(target_file)
        
        # Perform matching
        match_result = _fingerprinter.match_fingerprints(query_hashes, target_hashes)
        
        logger.info("Audio fingerprint matching completed", 
                   match=match_result.get("match", False),
                   confidence=match_result.get("confidence", 0.0))
        
        return match_result
        
    except Exception as e:
        logger.error("Audio fingerprint matching failed", 
                    query_file=query_file, target_file=target_file, error=str(e))
        return {"match": False, "error": str(e)}

def save_fingerprint_to_file(hashes: List[Tuple], output_path: str):
    """Save detailed fingerprint data to file."""
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(hashes, f)
        logger.info("Fingerprint saved to file", output_path=output_path, hash_count=len(hashes))
    except Exception as e:
        logger.error("Failed to save fingerprint", output_path=output_path, error=str(e))
        raise

def load_fingerprint_from_file(input_path: str) -> List[Tuple]:
    """Load detailed fingerprint data from file."""
    try:
        with open(input_path, 'rb') as f:
            hashes = pickle.load(f)
        logger.info("Fingerprint loaded from file", input_path=input_path, hash_count=len(hashes))
        return hashes
    except Exception as e:
        logger.error("Failed to load fingerprint", input_path=input_path, error=str(e))
        raise

def get_fingerprint_info() -> Dict:
    """Get information about the fingerprinting configuration."""
    return {
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "peak_threshold": PEAK_THRESHOLD,
        "fan_value": FAN_VALUE,
        "fingerprint_reduction": FINGERPRINT_REDUCTION,
        "algorithm": "spectral_peak_pair_hashing"
    }
