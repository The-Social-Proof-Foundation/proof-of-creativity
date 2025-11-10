"""
MySocial RPC Client for Proof of Creativity Oracle
Submits analysis results to MySocial blockchain
"""
import os
import json
import requests
import structlog
from typing import Optional
from app.services.mys_wallet import MySocialWallet, load_oracle_wallet

logger = structlog.get_logger()

class MySocialClient:
    """Client for submitting PoC analysis to MySocial blockchain"""
    
    def __init__(self, wallet: Optional[MySocialWallet] = None):
        """
        Initialize MySocial client
        
        Args:
            wallet: MySocialWallet instance (optional, will load from env if not provided)
        """
        # MySocial RPC configuration
        self.rpc_url = os.getenv("MYSOCIAL_RPC_URL", "https://fullnode.testnet.mysocial.io")
        
        # Contract addresses (from environment)
        self.package_id = os.getenv("MYS_POC_PACKAGE_ID")
        self.config_id = os.getenv("MYS_POC_CONFIG_ID")
        self.registry_id = os.getenv("MYS_POC_REGISTRY_ID")
        self.token_registry_id = os.getenv("MYS_TOKEN_REGISTRY_ID")
        
        # Wallet for signing
        self.wallet = wallet or load_oracle_wallet()
        
        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        logger.info("MySocial client initialized",
                   rpc_url=self.rpc_url,
                   oracle_address=self.wallet.get_address(),
                   package_id=self.package_id)
    
    def submit_poc_analysis(
        self,
        post_id: str,
        media_type: int,  # 1=image, 2=video, 3=audio
        similarity_score: int,  # 0-100
        original_creator: Optional[str] = None
    ) -> dict:
        """
        Submit PoC analysis result to MySocial blockchain
        
        Calls: proof_of_creativity::analyze_and_update_post
        
        Args:
            post_id: MySocial post object ID (0x...)
            media_type: 1=image, 2=video, 3=audio
            similarity_score: Highest similarity score (0-100)
            original_creator: Original creator address if derivative (0x...)
        
        Returns:
            Transaction result dict with tx_hash
        """
        try:
            # PRINT STATEMENT FOR JANE - Function entry
            print(f"\n[POC] submit_poc_analysis CALLED - post_id={post_id}, media_type={media_type}, similarity={similarity_score}\n")
            
            logger.info("Submitting PoC analysis to MySocial",
                       post_id=post_id,
                       media_type=media_type,
                       similarity_score=similarity_score,
                       is_derivative=original_creator is not None)
            
            # Build transaction data
            tx_data = self._build_analyze_transaction(
                post_id=post_id,
                media_type=media_type,
                similarity_score=similarity_score,
                original_creator=original_creator
            )
            
            # Sign transaction
            signature = self.wallet.sign_transaction(json.dumps(tx_data).encode())
            
            # Submit to RPC
            result = self._submit_transaction(tx_data, signature)
            
            logger.info("PoC analysis submitted successfully",
                       post_id=post_id,
                       tx_hash=result.get("tx_hash"),
                       status=result.get("status"))
            
            return result
            
        except Exception as e:
            logger.error("Failed to submit PoC analysis",
                        post_id=post_id,
                        error=str(e))
            raise
    
    def _build_analyze_transaction(
        self,
        post_id: str,
        media_type: int,
        similarity_score: int,
        original_creator: Optional[str]
    ) -> dict:
        """
        Build transaction for analyze_and_update_post call
        
        Transaction format for MySocial Move call:
        {
          "kind": "moveCall",
          "data": {
            "packageObjectId": "0x...",
            "module": "proof_of_creativity",
            "function": "analyze_and_update_post",
            "arguments": [...]
          }
        }
        """
        # Format original_creator as Option<address>
        original_creator_arg = [original_creator] if original_creator else []
        
        tx_data = {
            "kind": "moveCall",
            "data": {
                "packageObjectId": self.package_id,
                "module": "proof_of_creativity",
                "function": "analyze_and_update_post",
                "typeArguments": [],
                "arguments": [
                    self.config_id,          # config: &PoCConfig
                    self.registry_id,        # registry: &mut PoCRegistry
                    self.token_registry_id,  # token_registry: &TokenRegistry
                    post_id,                 # post: &mut Post
                    media_type,              # media_type: u8
                    similarity_score,        # highest_similarity_score: u64
                    original_creator_arg     # original_creator: Option<address>
                ]
            },
            "sender": self.wallet.get_address(),
            "gasBudget": "10000000",  # 0.01 MYS
            "gasPrice": "1000"
        }
        
        return tx_data
    
    def _submit_transaction(self, tx_data: dict, signature: str) -> dict:
        """
        Submit signed transaction to MySocial RPC
        
        Args:
            tx_data: Transaction data
            signature: Hex-encoded signature
            
        Returns:
            RPC response with tx_hash
        """
        # Build RPC request
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "mys_executeTransactionBlock",
            "params": [
                tx_data,
                [signature],
                {
                    "showInput": True,
                    "showEffects": True,
                    "showEvents": True
                }
            ]
        }
        
        # Extract transaction details for logging
        tx_call_data = tx_data.get("data", {})
        function_name = tx_call_data.get("function", "unknown")
        module_name = tx_call_data.get("module", "unknown")
        arguments = tx_call_data.get("arguments", [])
        
        # Parse arguments for detailed logging
        post_id = arguments[3] if len(arguments) > 3 else "unknown"
        media_type = arguments[4] if len(arguments) > 4 else "unknown"
        similarity_score = arguments[5] if len(arguments) > 5 else "unknown"
        original_creator = arguments[6] if len(arguments) > 6 else []
        
        # Map media type to human-readable
        media_type_map = {1: "image", 2: "video", 3: "audio"}
        media_type_name = media_type_map.get(media_type, f"unknown({media_type})")
        
        # PRINT STATEMENTS FOR JANE - These will show up regardless of logging config
        print("\n" + "=" * 80)
        print("🚀 SUBMITTING PROOF_OF_CREATIVITY TRANSACTION TO BLOCKCHAIN")
        print("=" * 80)
        print(f"Contract: {module_name}::{function_name}")
        print(f"Package ID: {self.package_id}")
        print(f"Post ID: {post_id}")
        print(f"Media Type: {media_type_name} ({media_type})")
        print(f"Similarity Score: {similarity_score}/100")
        print(f"Original Creator: {original_creator[0] if original_creator else 'None'}")
        print(f"Oracle Address: {self.wallet.get_address()}")
        print(f"RPC Endpoint: {self.rpc_url}")
        print("=" * 80)
        print("⏳ Submitting transaction to MySocial RPC...")
        print("=" * 80 + "\n")
        
        # Also log via structlog
        logger.info("🚀 SUBMITTING PROOF_OF_CREATIVITY TRANSACTION TO BLOCKCHAIN",
                   package_id=self.package_id,
                   module=module_name,
                   function=function_name,
                   post_id=post_id,
                   media_type=f"{media_type_name} ({media_type})",
                   similarity_score=f"{similarity_score}/100",
                   original_creator=original_creator[0] if original_creator else None,
                   oracle_address=self.wallet.get_address(),
                   rpc_endpoint=self.rpc_url)
        
        try:
            response = self.session.post(self.rpc_url, json=rpc_request, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if "error" in result:
                error_msg = result.get("error", {})
                # PRINT STATEMENTS FOR JANE - RPC Error
                print("\n" + "=" * 80)
                print("❌ PROOF_OF_CREATIVITY TRANSACTION REJECTED BY RPC")
                print("=" * 80)
                print(f"Error: {json.dumps(error_msg, indent=2)}")
                print("=" * 80 + "\n")
                logger.error("❌ PROOF_OF_CREATIVITY TRANSACTION REJECTED BY RPC",
                           error=json.dumps(error_msg, indent=2))
                raise Exception(f"RPC error: {error_msg}")
            
            # Extract transaction hash
            tx_result = result.get("result", {})
            tx_hash = tx_result.get("digest")
            tx_status = tx_result.get("effects", {}).get("status")
            tx_events = tx_result.get("events", [])
            
            # PRINT STATEMENTS FOR JANE - Success
            print("\n" + "=" * 80)
            print("✅ PROOF_OF_CREATIVITY TRANSACTION SUBMITTED SUCCESSFULLY")
            print("=" * 80)
            print(f"Transaction Hash: {tx_hash}")
            print(f"Status: {tx_status}")
            print(f"Events Count: {len(tx_events)}")
            if tx_events:
                print("\nTransaction Events:")
                for i, event in enumerate(tx_events):
                    event_type = event.get("type", "unknown")
                    print(f"  Event {i+1}: {event_type}")
            print("=" * 80 + "\n")
            
            # Also log via structlog
            logger.info("✅ PROOF_OF_CREATIVITY TRANSACTION SUBMITTED SUCCESSFULLY",
                       tx_hash=tx_hash,
                       status=tx_status,
                       events_count=len(tx_events))
            
            return {
                "success": True,
                "tx_hash": tx_hash,
                "status": tx_status,
                "events": tx_events
            }
            
        except requests.exceptions.RequestException as e:
            # PRINT STATEMENTS FOR JANE - Request Error
            print("\n" + "=" * 80)
            print("❌ PROOF_OF_CREATIVITY TRANSACTION SUBMISSION FAILED")
            print("=" * 80)
            print(f"RPC Request Error: {str(e)}")
            print(f"RPC URL: {self.rpc_url}")
            print(f"Error Type: {type(e).__name__}")
            print("=" * 80 + "\n")
            logger.error("❌ PROOF_OF_CREATIVITY TRANSACTION SUBMISSION FAILED",
                       error=str(e),
                       rpc_url=self.rpc_url,
                       error_type=type(e).__name__)
            raise
        except Exception as e:
            # PRINT STATEMENTS FOR JANE - General Error
            print("\n" + "=" * 80)
            print("❌ PROOF_OF_CREATIVITY TRANSACTION SUBMISSION FAILED")
            print("=" * 80)
            print(f"Transaction Error: {str(e)}")
            print(f"Error Type: {type(e).__name__}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text[:500]}")
            print("=" * 80 + "\n")
            logger.error("❌ PROOF_OF_CREATIVITY TRANSACTION SUBMISSION FAILED",
                       error=str(e),
                       error_type=type(e).__name__)
            raise
    
    def verify_oracle_authorization(self) -> bool:
        """Check if this wallet is authorized as oracle in PoCConfig"""
        try:
            # Query PoCConfig object
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "mys_getObject",
                "params": [
                    self.config_id,
                    {"showContent": True}
                ]
            }
            
            response = self.session.post(self.rpc_url, json=rpc_request, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            config_data = result.get("result", {}).get("data", {}).get("content", {}).get("fields", {})
            
            authorized_oracle = config_data.get("oracle_address")
            our_address = self.wallet.get_address()
            
            is_authorized = (authorized_oracle == our_address)
            
            if is_authorized:
                logger.info("✅ Oracle authorized",
                           our_address=our_address,
                           authorized_oracle=authorized_oracle)
            else:
                logger.error("❌ Oracle NOT authorized - ADDRESS MISMATCH",
                           our_address=our_address,
                           authorized_oracle=authorized_oracle,
                           note="The derived address doesn't match PoCConfig.oracle_address")
            
            return is_authorized
            
        except Exception as e:
            logger.error("Failed to verify oracle authorization", error=str(e))
            return False
    
    def check_post_already_analyzed(self, post_id: str) -> dict:
        """
        Check if a post has already been analyzed by PoC
        CRITICAL: Prevents duplicate analysis (one media per post rule)
        
        Returns:
            {
                "already_analyzed": bool,
                "poc_status": int or None,
                "poc_badge_id": str or None,
                "revenue_redirect_to": str or None
            }
        """
        try:
            # Query post object from MySocial
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "mys_getObject",
                "params": [
                    post_id,
                    {"showContent": True}
                ]
            }
            
            response = self.session.post(self.rpc_url, json=rpc_request, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            # Check for errors (post doesn't exist, etc.)
            if "error" in result:
                logger.warning("Post not found on blockchain", post_id=post_id)
                return {
                    "already_analyzed": False,
                    "poc_status": None,
                    "poc_badge_id": None,
                    "revenue_redirect_to": None,
                    "error": result["error"]
                }
            
            post_data = result.get("result", {}).get("data", {}).get("content", {}).get("fields", {})
            
            # Check PoC fields
            poc_status = post_data.get("poc_status")
            poc_badge_id = post_data.get("poc_badge_id")
            revenue_redirect_to = post_data.get("revenue_redirect_to")
            
            # If any PoC field is set, post was already analyzed
            already_analyzed = (
                poc_status is not None or
                poc_badge_id is not None or
                revenue_redirect_to is not None
            )
            
            logger.info("Post PoC check",
                       post_id=post_id,
                       already_analyzed=already_analyzed,
                       poc_status=poc_status)
            
            return {
                "already_analyzed": already_analyzed,
                "poc_status": poc_status,
                "poc_badge_id": poc_badge_id,
                "revenue_redirect_to": revenue_redirect_to
            }
            
        except Exception as e:
            logger.error("Failed to check post PoC status", post_id=post_id, error=str(e))
            # On error, assume not analyzed (fail open)
            return {
                "already_analyzed": False,
                "poc_status": None,
                "poc_badge_id": None,
                "revenue_redirect_to": None,
                "error": str(e)
            }
    
    def get_poc_config(self) -> dict:
        """Fetch current PoC configuration from blockchain"""
        try:
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "mys_getObject",
                "params": [
                    self.config_id,
                    {"showContent": True}
                ]
            }
            
            response = self.session.post(self.rpc_url, json=rpc_request, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            config_data = result.get("result", {}).get("data", {}).get("content", {}).get("fields", {})
            
            return {
                "oracle_address": config_data.get("oracle_address"),
                "image_threshold": int(config_data.get("image_threshold", 95)),
                "video_threshold": int(config_data.get("video_threshold", 95)),
                "audio_threshold": int(config_data.get("audio_threshold", 95)),
                "revenue_redirect_percentage": int(config_data.get("revenue_redirect_percentage", 100))
            }
            
        except Exception as e:
            logger.error("Failed to fetch PoC config", error=str(e))
            raise


def init_mys_client() -> Optional[MySocialClient]:
    """
    Initialize MySocial client if blockchain integration is enabled
    
    Returns MySocialClient or None if disabled
    """
    if os.getenv("MYS_INTEGRATION_ENABLED", "false").lower() != "true":
        logger.info("MySocial blockchain integration disabled")
        return None
    
    try:
        client = MySocialClient()
        
        # Verify oracle is authorized
        if client.verify_oracle_authorization():
            logger.info("✅ MySocial oracle authorized")
        else:
            logger.warning("⚠️  Oracle not authorized in PoCConfig - check MYS_ORACLE_PRIVATE_KEY")
        
        return client
        
    except Exception as e:
        logger.error("Failed to initialize MySocial client", error=str(e))
        logger.warning("Continuing without blockchain integration")
        return None

