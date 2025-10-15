import os
import structlog
from typing import BinaryIO, Optional, Dict, Any
from pathlib import Path
import time

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
import boto3
from botocore.exceptions import ClientError, BotoCoreError
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
    """Enhanced storage client with support for multiple backends: Cloudflare R2 (primary), GCS (backup), Walrus, and local."""
    
    def __init__(self):
        self.r2_client = None
        self.gcs_client = None
        self.session = None
        
        # Initialize Cloudflare R2 client if enabled (PRIMARY for videos/CDN)
        if config.USE_CLOUDFLARE_R2:
            try:
                self._initialize_r2()
            except Exception as e:
                logger.error("Failed to initialize Cloudflare R2 client", error=str(e))
                logger.warning("R2 initialization failed, will try other storage backends")
        
        # Initialize GCS client if enabled (BACKUP/ARCHIVE - optional)
        if config.USE_GCS:
            try:
                self._initialize_gcs()
            except Exception as e:
                logger.error("Failed to initialize GCS client", error=str(e))
                logger.warning("GCS initialization failed, continuing without GCS storage")
        
        # Initialize HTTP session for Walrus
        if config.USE_WALRUS:
            try:
                self._initialize_walrus_session()
            except Exception as e:
                logger.error("Failed to initialize Walrus client", error=str(e))
                logger.warning("Walrus initialization failed, continuing without Walrus storage")
        
        # Check if any storage backend is available
        has_storage = (
            self.r2_client is not None or 
            self.gcs_client is not None or 
            (config.USE_WALRUS and self.session is not None)
        )
        if not has_storage:
            logger.warning("No storage backends available - uploads will use local storage only")
        
        logger.info("Storage client initialized", 
                   r2_enabled=self.r2_client is not None,
                   gcs_enabled=self.gcs_client is not None,
                   walrus_enabled=config.USE_WALRUS and self.session is not None)
    
    def _initialize_r2(self):
        """Initialize Cloudflare R2 client (S3-compatible)."""
        try:
            # Get R2 credentials from environment
            account_id = os.getenv("R2_ACCOUNT_ID")
            access_key_id = os.getenv("R2_ACCESS_KEY_ID")
            secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
            
            if not all([account_id, access_key_id, secret_access_key]):
                raise ValueError("Missing R2 credentials: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, or R2_SECRET_ACCESS_KEY")
            
            # R2 endpoint format: https://<account_id>.r2.cloudflarestorage.com
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
            
            # Create boto3 S3 client configured for R2
            self.r2_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name='auto'  # R2 uses 'auto' region
            )
            
            # Test bucket access
            bucket_name = config.R2_BUCKET_NAME
            try:
                self.r2_client.head_bucket(Bucket=bucket_name)
                logger.info("Cloudflare R2 client initialized successfully", 
                           bucket_name=bucket_name,
                           account_id=account_id)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == '404':
                    logger.warning("R2 bucket does not exist", bucket_name=bucket_name)
                    raise ValueError(f"R2 bucket does not exist: {bucket_name}")
                else:
                    logger.error("R2 bucket access test failed", error=str(e), error_code=error_code)
                    raise
                    
        except Exception as e:
            logger.error("R2 initialization failed", error=str(e))
            raise
    
    def _initialize_gcs(self):
        """Initialize Google Cloud Storage client."""
        try:
            # Option 1: Check for JSON credentials in environment variable (Railway-friendly)
            gcs_json = os.getenv("GCS_SERVICE_ACCOUNT_JSON")
            if gcs_json:
                import json
                import tempfile
                from google.oauth2 import service_account
                
                logger.info("Loading GCS credentials from environment variable")
                
                try:
                    # Parse JSON credentials
                    credentials_info = json.loads(gcs_json)
                    credentials = service_account.Credentials.from_service_account_info(credentials_info)
                    self.gcs_client = storage.Client(credentials=credentials, project=credentials_info.get('project_id'))
                    logger.info("GCS client created from environment JSON credentials")
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse GCS_SERVICE_ACCOUNT_JSON", error=str(e))
                    raise ValueError("Invalid JSON in GCS_SERVICE_ACCOUNT_JSON environment variable")
            
            # Option 2: Check for credentials file path (traditional method)
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if not os.path.exists(credentials_path):
                    logger.warning("GCS credentials file not found", path=credentials_path)
                    raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
                
                logger.info("Loading GCS credentials from file", path=credentials_path)
                self.gcs_client = storage.Client()
            
            # Option 3: Try default credentials (for GCE, Cloud Run, etc.)
            else:
                logger.info("Attempting to use default GCS credentials")
                self.gcs_client = storage.Client()
            
            # Test bucket access
            bucket = self.gcs_client.bucket(config.GCS_BUCKET_NAME)
            if not bucket.exists():
                logger.warning("GCS bucket does not exist", bucket_name=config.GCS_BUCKET_NAME)
                raise ValueError(f"GCS bucket does not exist: {config.GCS_BUCKET_NAME}")
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
    
    def upload(self, filename: str, fileobj: BinaryIO, media_type: str = "unknown", progress_callback: Optional[callable] = None) -> str:
        """
        Upload file to configured storage backend(s) with enhanced error handling.
        Priority: R2 (primary) → GCS (optional backup) → Walrus → Local
        
        Args:
            filename: Name of the file
            fileobj: File object to upload
            media_type: Type of media (image, audio, video)
            
        Returns:
            Dict with 'primary' storage URI and optional 'backup' URI
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
        
        storage_uris = {}
        
        # Try Cloudflare R2 first (PRIMARY for CDN delivery)
        if config.USE_CLOUDFLARE_R2 and self.r2_client:
            try:
                r2_uri = self._upload_to_r2(filename, fileobj, media_type, file_size, progress_callback)
                storage_uris['primary'] = r2_uri
                logger.info("Primary storage (R2) upload successful", uri=r2_uri)
                
                # Optionally backup to GCS if enabled
                if config.USE_GCS and config.GCS_BACKUP_ENABLED and self.gcs_client:
                    try:
                        fileobj.seek(0)  # Reset for second upload
                        gcs_uri = self._upload_to_gcs(filename, fileobj, media_type, file_size)
                        storage_uris['backup'] = gcs_uri
                        logger.info("Backup storage (GCS) upload successful", uri=gcs_uri)
                    except Exception as e:
                        logger.warning("GCS backup upload failed, continuing", error=str(e))
                
                return storage_uris['primary']  # Return primary URI for backward compatibility
                
            except Exception as e:
                logger.error("R2 upload failed", filename=filename, error=str(e))
                logger.info("Falling back to other storage backends")
        
        # Try GCS if R2 failed or not configured
        if config.USE_GCS and self.gcs_client:
            try:
                gcs_uri = self._upload_to_gcs(filename, fileobj, media_type, file_size)
                storage_uris['primary'] = gcs_uri
                return gcs_uri
            except Exception as e:
                logger.error("GCS upload failed", filename=filename, error=str(e))
                if not config.USE_WALRUS:
                    logger.warning("Falling back to local storage")
                    return self._upload_to_local(filename, fileobj, media_type, file_size)
                logger.info("Falling back to Walrus storage")
        
        # Try Walrus if enabled
        if config.USE_WALRUS and self.session:
            try:
                return self._upload_to_walrus(filename, fileobj, media_type, file_size)
            except Exception as e:
                logger.error("Walrus upload failed", filename=filename, error=str(e))
                logger.warning("All cloud storage backends failed, falling back to local storage")
                return self._upload_to_local(filename, fileobj, media_type, file_size)
        
        # Fallback to local storage if no backends are configured
        logger.warning("No cloud storage backends available, using local storage")
        return self._upload_to_local(filename, fileobj, media_type, file_size)
    
    def _upload_to_r2(self, filename: str, fileobj: BinaryIO, media_type: str, file_size: int, progress_callback: Optional[callable] = None) -> str:
        """Upload file to Cloudflare R2 (S3-compatible) with progress tracking."""
        try:
            bucket_name = config.R2_BUCKET_NAME
            
            # Simplified storage path: media_type/YYYY/MM/filename (filename is just media_id.ext)
            from datetime import datetime
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m")
            storage_path = f"{media_type}/{year}/{month}/{filename}"
            
            # Set metadata (sanitize to ASCII-only for S3/R2 compatibility)
            metadata = {
                "original_filename": filename.encode('ascii', 'ignore').decode('ascii'),
                "media_type": media_type,
                "file_size": str(file_size),
                "upload_timestamp": str(int(time.time()))
            }
            
            # Set content type
            content_type = self._get_content_type(filename) or 'application/octet-stream'
            
            # Upload with progress tracking for large files
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.info("Uploading large file to R2", filename=filename, file_size_mb=file_size/1024/1024)
            
            start_time = time.time()
            
            # Upload to R2 using boto3 S3 client with progress tracking
            fileobj.seek(0)  # Reset to beginning
            
            if progress_callback and file_size > 0:
                # Wrap file object to track bytes uploaded
                from app.core.utils import ProgressFileWrapper
                wrapped_fileobj = ProgressFileWrapper(fileobj, file_size, progress_callback)
                
                self.r2_client.put_object(
                    Bucket=bucket_name,
                    Key=storage_path,
                    Body=wrapped_fileobj,
                    ContentType=content_type,
                    Metadata=metadata
                )
            else:
                # No progress tracking
                self.r2_client.put_object(
                    Bucket=bucket_name,
                    Key=storage_path,
                    Body=fileobj,
                    ContentType=content_type,
                    Metadata=metadata
                )
            
            upload_time = time.time() - start_time
            
            # Generate storage URI
            storage_uri = f"r2://{bucket_name}/{storage_path}"
            
            # Generate CDN URL if custom domain is configured
            cdn_url = None
            if config.R2_PUBLIC_DOMAIN:
                cdn_url = f"https://{config.R2_PUBLIC_DOMAIN}/{storage_path}"
                logger.info("R2 upload completed with CDN URL", 
                           filename=filename,
                           storage_uri=storage_uri,
                           cdn_url=cdn_url,
                           upload_time_seconds=round(upload_time, 2),
                           upload_speed_mbps=round((file_size / 1024 / 1024) / upload_time, 2) if upload_time > 0 else 0)
            else:
                logger.info("R2 upload completed successfully", 
                           filename=filename,
                           storage_uri=storage_uri,
                           upload_time_seconds=round(upload_time, 2),
                           upload_speed_mbps=round((file_size / 1024 / 1024) / upload_time, 2) if upload_time > 0 else 0)
            
            return storage_uri
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error("R2 API error during upload", 
                        filename=filename, error=str(e), error_code=error_code)
            raise
        except Exception as e:
            logger.error("Unexpected error during R2 upload", filename=filename, error=str(e))
            raise
    
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
            
            # Set metadata (sanitize to ASCII-only for consistency)
            blob.metadata = {
                "original_filename": filename.encode('ascii', 'ignore').decode('ascii'),
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
            storage_uri: Storage URI (r2://, gs://, walrus://, or local://)
            output_path: Optional path to save file
            
        Returns:
            Path to downloaded file
        """
        logger.info("Starting file download", storage_uri=storage_uri)
        
        if storage_uri.startswith("r2://"):
            return self._download_from_r2(storage_uri, output_path)
        elif storage_uri.startswith("gs://"):
            return self._download_from_gcs(storage_uri, output_path)
        elif storage_uri.startswith("walrus://"):
            return self._download_from_walrus(storage_uri, output_path)
        elif storage_uri.startswith("local://"):
            # Local files can be accessed directly
            local_path = storage_uri.replace("local://", "")
            return local_path if os.path.exists(local_path) else None
        else:
            raise StorageError(f"Unsupported storage URI format: {storage_uri}")
    
    def _download_from_r2(self, storage_uri: str, output_path: Optional[str] = None) -> str:
        """Download file from Cloudflare R2."""
        try:
            # Parse R2 URI: r2://bucket-name/path/to/file
            if not storage_uri.startswith("r2://"):
                raise ValueError("Invalid R2 URI")
            
            path_parts = storage_uri[5:].split("/", 1)  # Remove r2://
            bucket_name = path_parts[0]
            object_key = path_parts[1] if len(path_parts) > 1 else ""
            
            # Check if object exists
            try:
                self.r2_client.head_object(Bucket=bucket_name, Key=object_key)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == '404':
                    raise StorageError(f"File not found in R2: {storage_uri}")
                raise
            
            # Determine output path
            if not output_path:
                output_path = f"/tmp/{Path(object_key).name}"
            
            # Download file
            self.r2_client.download_file(bucket_name, object_key, output_path)
            
            logger.info("R2 download completed", 
                       storage_uri=storage_uri, output_path=output_path)
            
            return output_path
            
        except ClientError as e:
            logger.error("R2 download failed", storage_uri=storage_uri, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error during R2 download", storage_uri=storage_uri, error=str(e))
            raise
    
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
            "r2": {"available": False, "error": None},
            "gcs": {"available": False, "error": None},
            "walrus": {"available": False, "error": None}
        }
        
        # Check Cloudflare R2
        if config.USE_CLOUDFLARE_R2 and self.r2_client:
            try:
                self.r2_client.head_bucket(Bucket=config.R2_BUCKET_NAME)
                health["r2"]["available"] = True
                health["r2"]["bucket"] = config.R2_BUCKET_NAME
                if config.R2_PUBLIC_DOMAIN:
                    health["r2"]["cdn_domain"] = config.R2_PUBLIC_DOMAIN
            except ClientError as e:
                health["r2"]["error"] = str(e)
            except Exception as e:
                health["r2"]["error"] = str(e)
        
        # Check GCS (backup)
        if config.USE_GCS and self.gcs_client:
            try:
                bucket = self.gcs_client.bucket(config.GCS_BUCKET_NAME)
                bucket.exists()  # This will test connectivity
                health["gcs"]["available"] = True
                health["gcs"]["backup_mode"] = config.GCS_BACKUP_ENABLED
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
