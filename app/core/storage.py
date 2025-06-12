import os
import structlog
from typing import BinaryIO, Optional, Dict, Any
from pathlib import Path
import time

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app import config
from app.core.utils import create_media_storage_path, format_file_size

logger = structlog.get_logger()

class StorageError(Exception):
    """Custom exception for storage operations."""
    pass

class StorageClient:
    """Enhanced storage client with support for multiple backends."""
    
    def __init__(self):
        self.gcs_client = None
        self.session = None
        
        # Initialize GCS client if enabled
        if config.USE_GCS:
            try:
                self._initialize_gcs()
            except Exception as e:
                logger.error("Failed to initialize GCS client", error=str(e))
                # Don't fail startup for storage issues in development
                logger.warning("GCS initialization failed, continuing without GCS storage")
        
        # Initialize HTTP session for Walrus
        if config.USE_WALRUS:
            try:
                self._initialize_walrus_session()
            except Exception as e:
                logger.error("Failed to initialize Walrus client", error=str(e))
                logger.warning("Walrus initialization failed, continuing without Walrus storage")
        
        # Check if any storage backend is available
        has_storage = (self.gcs_client is not None) or (config.USE_WALRUS and self.session is not None)
        if not has_storage:
            logger.warning("No storage backends available - uploads will use local storage only")
        
        logger.info("Storage client initialized", 
                   gcs_enabled=self.gcs_client is not None,
                   walrus_enabled=config.USE_WALRUS and self.session is not None)
    
    def _initialize_gcs(self):
        """Initialize Google Cloud Storage client."""
        try:
            # Check for credentials
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path and not os.path.exists(credentials_path):
                logger.warning("GCS credentials file not found", path=credentials_path)
            
            self.gcs_client = storage.Client()
            
            # Test bucket access
            bucket = self.gcs_client.bucket(config.GCS_BUCKET_NAME)
            if not bucket.exists():
                logger.warning("GCS bucket does not exist", bucket_name=config.GCS_BUCKET_NAME)
            else:
                logger.info("GCS client initialized successfully", 
                           bucket_name=config.GCS_BUCKET_NAME)
                           
        except Exception as e:
            logger.error("GCS initialization failed", error=str(e))
            raise
    
    def _initialize_walrus_session(self):
        """Initialize HTTP session for Walrus with retry logic."""
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set timeout
        self.session.timeout = 30
        
        logger.info("Walrus HTTP session initialized", endpoint=config.WALRUS_ENDPOINT)
    
    def upload(self, filename: str, fileobj: BinaryIO, media_type: str = "unknown") -> str:
        """
        Upload file to configured storage backend with enhanced error handling.
        
        Args:
            filename: Name of the file
            fileobj: File object to upload
            media_type: Type of media (image, audio, video)
            
        Returns:
            Storage URI where the file was uploaded
        """
        # Get file size for logging
        current_pos = fileobj.tell()
        fileobj.seek(0, 2)  # Seek to end
        file_size = fileobj.tell()
        fileobj.seek(current_pos)  # Reset position
        
        logger.info("Starting file upload", 
                   filename=filename, 
                   file_size_human=format_file_size(file_size),
                   media_type=media_type)
        
        # Try GCS first if enabled
        if config.USE_GCS and self.gcs_client:
            try:
                return self._upload_to_gcs(filename, fileobj, media_type, file_size)
            except Exception as e:
                logger.error("GCS upload failed", filename=filename, error=str(e))
                if not config.USE_WALRUS:
                    raise StorageError(f"GCS upload failed: {e}")
                logger.info("Falling back to Walrus storage")
        
        # Try Walrus if enabled
        if config.USE_WALRUS and self.session:
            try:
                return self._upload_to_walrus(filename, fileobj, media_type, file_size)
            except Exception as e:
                logger.error("Walrus upload failed", filename=filename, error=str(e))
                if not self.gcs_client:
                    logger.warning("All storage backends failed, falling back to local storage")
                    return self._upload_to_local(filename, fileobj, media_type, file_size)
                raise StorageError(f"All storage backends failed. Last error: {e}")
        
        # Fallback to local storage if no backends are configured
        logger.warning("No storage backends available, using local storage")
        return self._upload_to_local(filename, fileobj, media_type, file_size)
    
    def _upload_to_gcs(self, filename: str, fileobj: BinaryIO, media_type: str, file_size: int) -> str:
        """Upload file to Google Cloud Storage."""
        try:
            bucket = self.gcs_client.bucket(config.GCS_BUCKET_NAME)
            
            # Create structured storage path
            storage_path = create_media_storage_path(
                media_id="", # Will be handled by caller
                filename=filename,
                media_type=media_type
            )
            
            blob = bucket.blob(storage_path)
            
            # Set metadata
            blob.metadata = {
                "original_filename": filename,
                "media_type": media_type,
                "file_size": str(file_size),
                "upload_timestamp": str(int(time.time()))
            }
            
            # Set content type
            content_type = self._get_content_type(filename)
            if content_type:
                blob.content_type = content_type
            
            # Upload with progress tracking for large files
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.info("Uploading large file", filename=filename, file_size_mb=file_size/1024/1024)
            
            start_time = time.time()
            blob.upload_from_file(fileobj, rewind=True)
            upload_time = time.time() - start_time
            
            storage_uri = f"gs://{config.GCS_BUCKET_NAME}/{storage_path}"
            
            logger.info("GCS upload completed successfully", 
                       filename=filename,
                       storage_uri=storage_uri,
                       upload_time_seconds=round(upload_time, 2),
                       upload_speed_mbps=round((file_size / 1024 / 1024) / upload_time, 2) if upload_time > 0 else 0)
            
            return storage_uri
            
        except GoogleCloudError as e:
            logger.error("GCS API error during upload", 
                        filename=filename, error=str(e), error_code=getattr(e, 'code', None))
            raise
        except Exception as e:
            logger.error("Unexpected error during GCS upload", filename=filename, error=str(e))
            raise
    
    def _upload_to_walrus(self, filename: str, fileobj: BinaryIO, media_type: str, file_size: int) -> str:
        """Upload file to Walrus storage."""
        try:
            # Create upload endpoint
            upload_endpoint = f"{config.WALRUS_ENDPOINT}/upload/{filename}"
            
            # Prepare headers
            headers = {
                'Content-Type': self._get_content_type(filename) or 'application/octet-stream',
                'X-Media-Type': media_type,
                'X-File-Size': str(file_size)
            }
            
            start_time = time.time()
            
            # Upload file
            response = self.session.put(
                upload_endpoint,
                data=fileobj,
                headers=headers,
                timeout=300  # 5 minutes for large files
            )
            
            upload_time = time.time() - start_time
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            cid = response_data.get("cid")
            
            if not cid:
                raise StorageError("Walrus upload succeeded but no CID returned")
            
            storage_uri = f"walrus://{cid}"
            
            logger.info("Walrus upload completed successfully", 
                       filename=filename,
                       storage_uri=storage_uri,
                       cid=cid,
                       upload_time_seconds=round(upload_time, 2))
            
            return storage_uri
            
        except requests.exceptions.RequestException as e:
            logger.error("Walrus HTTP error during upload", 
                        filename=filename, error=str(e), status_code=getattr(e.response, 'status_code', None))
            raise
        except Exception as e:
            logger.error("Unexpected error during Walrus upload", filename=filename, error=str(e))
            raise
    
    def _upload_to_local(self, filename: str, fileobj: BinaryIO, media_type: str, file_size: int) -> str:
        """Upload file to local storage as fallback."""
        try:
            # Create local uploads directory
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            # Create subdirectory by media type
            media_dir = uploads_dir / media_type
            media_dir.mkdir(exist_ok=True)
            
            # Create unique filename with timestamp
            timestamp = int(time.time())
            local_filename = f"{timestamp}_{filename}"
            local_path = media_dir / local_filename
            
            # Save file
            start_time = time.time()
            with open(local_path, 'wb') as f:
                fileobj.seek(0)  # Reset to beginning
                while True:
                    chunk = fileobj.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            
            upload_time = time.time() - start_time
            storage_uri = f"local://{local_path}"
            
            logger.info("Local upload completed successfully", 
                       filename=filename,
                       storage_uri=storage_uri,
                       upload_time_seconds=round(upload_time, 2))
            
            return storage_uri
            
        except Exception as e:
            logger.error("Unexpected error during local upload", filename=filename, error=str(e))
            raise
    
    def download(self, storage_uri: str, output_path: Optional[str] = None) -> str:
        """
        Download file from storage URI.
        
        Args:
            storage_uri: Storage URI (gs:// or walrus://)
            output_path: Optional path to save file
            
        Returns:
            Path to downloaded file
        """
        logger.info("Starting file download", storage_uri=storage_uri)
        
        if storage_uri.startswith("gs://"):
            return self._download_from_gcs(storage_uri, output_path)
        elif storage_uri.startswith("walrus://"):
            return self._download_from_walrus(storage_uri, output_path)
        else:
            raise StorageError(f"Unsupported storage URI format: {storage_uri}")
    
    def _download_from_gcs(self, storage_uri: str, output_path: Optional[str] = None) -> str:
        """Download file from Google Cloud Storage."""
        try:
            # Parse GCS URI
            if not storage_uri.startswith("gs://"):
                raise ValueError("Invalid GCS URI")
            
            path_parts = storage_uri[5:].split("/", 1)  # Remove gs://
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""
            
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                raise StorageError(f"File not found in GCS: {storage_uri}")
            
            # Determine output path
            if not output_path:
                output_path = f"/tmp/{Path(blob_path).name}"
            
            # Download file
            blob.download_to_filename(output_path)
            
            logger.info("GCS download completed", 
                       storage_uri=storage_uri, output_path=output_path)
            
            return output_path
            
        except Exception as e:
            logger.error("GCS download failed", storage_uri=storage_uri, error=str(e))
            raise
    
    def _download_from_walrus(self, storage_uri: str, output_path: Optional[str] = None) -> str:
        """Download file from Walrus storage."""
        try:
            # Parse Walrus URI
            if not storage_uri.startswith("walrus://"):
                raise ValueError("Invalid Walrus URI")
            
            cid = storage_uri[9:]  # Remove walrus://
            download_endpoint = f"{config.WALRUS_ENDPOINT}/download/{cid}"
            
            # Determine output path
            if not output_path:
                output_path = f"/tmp/walrus_{cid}"
            
            # Download file
            response = self.session.get(download_endpoint, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Walrus download completed", 
                       storage_uri=storage_uri, output_path=output_path)
            
            return output_path
            
        except Exception as e:
            logger.error("Walrus download failed", storage_uri=storage_uri, error=str(e))
            raise
    
    def delete(self, storage_uri: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            storage_uri: Storage URI to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Deleting file", storage_uri=storage_uri)
            
            if storage_uri.startswith("gs://"):
                return self._delete_from_gcs(storage_uri)
            elif storage_uri.startswith("walrus://"):
                return self._delete_from_walrus(storage_uri)
            else:
                logger.error("Unsupported storage URI for deletion", storage_uri=storage_uri)
                return False
                
        except Exception as e:
            logger.error("File deletion failed", storage_uri=storage_uri, error=str(e))
            return False
    
    def _delete_from_gcs(self, storage_uri: str) -> bool:
        """Delete file from Google Cloud Storage."""
        try:
            path_parts = storage_uri[5:].split("/", 1)  # Remove gs://
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""
            
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                blob.delete()
                logger.info("GCS file deleted successfully", storage_uri=storage_uri)
                return True
            else:
                logger.warning("GCS file not found for deletion", storage_uri=storage_uri)
                return True  # Consider it successful if file doesn't exist
                
        except Exception as e:
            logger.error("GCS deletion failed", storage_uri=storage_uri, error=str(e))
            return False
    
    def _delete_from_walrus(self, storage_uri: str) -> bool:
        """Delete file from Walrus storage."""
        try:
            cid = storage_uri[9:]  # Remove walrus://
            delete_endpoint = f"{config.WALRUS_ENDPOINT}/delete/{cid}"
            
            response = self.session.delete(delete_endpoint, timeout=30)
            
            if response.status_code == 204 or response.status_code == 404:
                logger.info("Walrus file deleted successfully", storage_uri=storage_uri)
                return True
            else:
                logger.error("Walrus deletion failed", 
                           storage_uri=storage_uri, status_code=response.status_code)
                return False
                
        except Exception as e:
            logger.error("Walrus deletion failed", storage_uri=storage_uri, error=str(e))
            return False
    
    def get_file_info(self, storage_uri: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from storage."""
        try:
            if storage_uri.startswith("gs://"):
                return self._get_gcs_file_info(storage_uri)
            elif storage_uri.startswith("walrus://"):
                return self._get_walrus_file_info(storage_uri)
            else:
                return None
                
        except Exception as e:
            logger.error("Failed to get file info", storage_uri=storage_uri, error=str(e))
            return None
    
    def _get_gcs_file_info(self, storage_uri: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from GCS."""
        try:
            path_parts = storage_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""
            
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                return None
            
            blob.reload()
            
            return {
                "size": blob.size,
                "content_type": blob.content_type,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "etag": blob.etag,
                "metadata": blob.metadata or {}
            }
            
        except Exception as e:
            logger.error("Failed to get GCS file info", storage_uri=storage_uri, error=str(e))
            return None
    
    def _get_walrus_file_info(self, storage_uri: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from Walrus."""
        try:
            cid = storage_uri[9:]
            info_endpoint = f"{config.WALRUS_ENDPOINT}/info/{cid}"
            
            response = self.session.get(info_endpoint, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error("Failed to get Walrus file info", storage_uri=storage_uri, error=str(e))
            return None
    
    def _get_content_type(self, filename: str) -> Optional[str]:
        """Get content type for filename."""
        import mimetypes
        content_type, _ = mimetypes.guess_type(filename)
        return content_type
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of storage backends."""
        health = {
            "gcs": {"available": False, "error": None},
            "walrus": {"available": False, "error": None}
        }
        
        # Check GCS
        if config.USE_GCS and self.gcs_client:
            try:
                bucket = self.gcs_client.bucket(config.GCS_BUCKET_NAME)
                bucket.exists()  # This will test connectivity
                health["gcs"]["available"] = True
            except Exception as e:
                health["gcs"]["error"] = str(e)
        
        # Check Walrus
        if config.USE_WALRUS and self.session:
            try:
                response = self.session.get(f"{config.WALRUS_ENDPOINT}/health", timeout=5)
                if response.status_code == 200:
                    health["walrus"]["available"] = True
                else:
                    health["walrus"]["error"] = f"HTTP {response.status_code}"
            except Exception as e:
                health["walrus"]["error"] = str(e)
        
        return health
