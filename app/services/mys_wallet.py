"""
MySocial Wallet and Transaction Signing
Supports MySocial-specific derivation paths and transaction signing
"""
import os
import hashlib
import hmac
import struct
from typing import Optional, Union
import structlog
from eth_account import Account
from ecdsa import SigningKey, SECP256k1, NIST256p
from ecdsa.util import sigencode_string
from bip_utils import Bip32Slip10Secp256k1, Bip32Slip10Nist256p1, Bip32Slip10Ed25519, Bip39SeedGenerator, Bip39MnemonicValidator
from nacl.signing import SigningKey as Ed25519SigningKey
from nacl.encoding import RawEncoder

logger = structlog.get_logger()

class MySocialWallet:
    """
    MySocial wallet with custom derivation paths
    - ed25519: m/44'/6976'/{account}'/{change}'/{address}'
    - secp256k1: m/54'/6976'/{account}'/{change}/{address}
    - secp256r1: m/74'/6976'/{account}'/{change}/{address}
    """
    
    def __init__(self, mnemonic: Optional[str] = None, private_key: Optional[str] = None, curve: str = "ed25519"):
        """
        Initialize wallet from mnemonic or private key
        
        Args:
            mnemonic: BIP39 mnemonic phrase
            private_key: Hex-encoded private key
            curve: "secp256k1" or "secp256r1"
        """
        self.curve = curve
        self.private_key_hex = None
        self.address = None
        
        if private_key:
            # Handle different MySocial private key formats
            if private_key.startswith("0x"):
                # Already hex
                self.private_key_hex = private_key
            elif all(c in '0123456789abcdefABCDEF' for c in private_key):
                # Hex without 0x prefix
                self.private_key_hex = f"0x{private_key}"
            elif private_key.startswith("mysoprivkey"):
                # Bech32 format (MySocial wallet export)
                raise ValueError("Bech32 'mysoprivkey' format not yet supported. Please use raw hex or Base64.")
            else:
                # Try Base64 decoding (common MySocial export format)
                try:
                    import base64
                    decoded = base64.b64decode(private_key)
                    # MySocial keys: 1-byte flag (0x00, 0x01, 0x02) + 32-byte private key
                    if len(decoded) == 33:
                        flag = decoded[0]
                        private_key_bytes = decoded[1:]  # Remove flag byte
                        self.private_key_hex = "0x" + private_key_bytes.hex()
                        logger.debug("Decoded Base64 key", flag=flag, curve=curve)
                    elif len(decoded) == 32:
                        # Raw 32-byte key (no flag)
                        self.private_key_hex = "0x" + decoded.hex()
                    else:
                        raise ValueError(f"Unexpected key length: {len(decoded)} bytes")
                except Exception as e:
                    raise ValueError(f"Invalid private key format (expected hex or Base64): {e}")
            
            self._derive_address()
            logger.info("MySocial wallet initialized from private key", curve=curve)
            
        elif mnemonic:
            # Derive from mnemonic with MySocial path
            self._init_from_mnemonic(mnemonic, account=0, change=0, address_index=0)
            logger.info("MySocial wallet initialized from mnemonic", curve=curve)
            
        else:
            raise ValueError("Must provide either mnemonic or private_key")
    
    def _init_from_mnemonic(self, mnemonic: str, account: int = 0, change: int = 0, address_index: int = 0):
        """Initialize wallet from mnemonic with MySocial derivation path"""
        
        # Validate mnemonic
        if not Bip39MnemonicValidator().IsValid(mnemonic):
            raise ValueError("Invalid BIP39 mnemonic")
        
        # Generate seed
        seed = Bip39SeedGenerator(mnemonic).Generate()
        
        if self.curve == "ed25519":
            # MySocial Ed25519 path: m/44'/6976'/{account}'/{change}'/{address}' (all hardened)
            bip32_ctx = Bip32Slip10Ed25519.FromSeed(seed)
            
            # Derive following MySocial path
            bip32_ctx = bip32_ctx.ChildKey(44 + 0x80000000)  # 44' (hardened)
            bip32_ctx = bip32_ctx.ChildKey(6976 + 0x80000000)  # 6976' (hardened) - MySocial coin type
            bip32_ctx = bip32_ctx.ChildKey(account + 0x80000000)  # account' (hardened)
            bip32_ctx = bip32_ctx.ChildKey(change + 0x80000000)  # change' (hardened)
            bip32_ctx = bip32_ctx.ChildKey(address_index + 0x80000000)  # address' (hardened)
            
        elif self.curve == "secp256k1":
            # MySocial secp256k1 path: m/54'/6976'/{account}'/{change}/{address}
            bip32_ctx = Bip32Slip10Secp256k1.FromSeed(seed)
            
            # Derive following MySocial path
            bip32_ctx = bip32_ctx.ChildKey(54 + 0x80000000)  # 54' (hardened)
            bip32_ctx = bip32_ctx.ChildKey(6976 + 0x80000000)  # 6976' (hardened) - MySocial coin type
            bip32_ctx = bip32_ctx.ChildKey(account + 0x80000000)  # account' (hardened)
            bip32_ctx = bip32_ctx.ChildKey(change)  # change (normal)
            bip32_ctx = bip32_ctx.ChildKey(address_index)  # address (normal)
            
        elif self.curve == "secp256r1":
            # MySocial secp256r1 path: m/74'/6976'/{account}'/{change}/{address}
            bip32_ctx = Bip32Slip10Nist256p1.FromSeed(seed)
            
            # Derive following MySocial path
            bip32_ctx = bip32_ctx.ChildKey(74 + 0x80000000)  # 74' (hardened)
            bip32_ctx = bip32_ctx.ChildKey(6976 + 0x80000000)  # 6976' (hardened) - MySocial coin type
            bip32_ctx = bip32_ctx.ChildKey(account + 0x80000000)  # account' (hardened)
            bip32_ctx = bip32_ctx.ChildKey(change)  # change (normal)
            bip32_ctx = bip32_ctx.ChildKey(address_index)  # address (normal)
        else:
            raise ValueError(f"Unsupported curve: {self.curve}")
        
        # Extract private key
        self.private_key_hex = bip32_ctx.PrivateKey().Raw().ToHex()
        self._derive_address()
    
    def _derive_address(self):
        """Derive MySocial address from private key using BLAKE2b-256"""
        if self.curve == "ed25519":
            # Get public key from Ed25519 private key
            private_key_bytes = bytes.fromhex(self.private_key_hex.replace("0x", ""))
            signing_key = Ed25519SigningKey(private_key_bytes)
            verify_key = signing_key.verify_key
            pub_key_bytes = bytes(verify_key)  # 32 bytes
            
            # MySocial address: BLAKE2b(0x00 || pub_key_bytes)
            flag_and_pubkey = bytes([0x00]) + pub_key_bytes  # 0x00 = Ed25519 flag
            address_bytes = hashlib.blake2b(flag_and_pubkey, digest_size=32).digest()
            self.address = "0x" + address_bytes.hex()
            
        elif self.curve == "secp256k1":
            # Get public key using eth_account
            account = Account.from_key(self.private_key_hex)
            # Get uncompressed public key (65 bytes: 0x04 + 32-byte x + 32-byte y)
            pub_key_bytes = account._key_obj.public_key.to_bytes()
            
            # MySocial address: BLAKE2b(0x01 || pub_key_bytes)
            flag_and_pubkey = bytes([0x01]) + pub_key_bytes  # 0x01 = secp256k1 flag
            address_bytes = hashlib.blake2b(flag_and_pubkey, digest_size=32).digest()
            self.address = "0x" + address_bytes.hex()
            
        elif self.curve == "secp256r1":
            # Get public key using ecdsa library
            private_key_bytes = bytes.fromhex(self.private_key_hex.replace("0x", ""))
            signing_key = SigningKey.from_string(private_key_bytes, curve=NIST256p)
            verifying_key = signing_key.get_verifying_key()
            
            # Get uncompressed public key (64 bytes: 32-byte x + 32-byte y)
            pub_key_bytes = b'\x04' + verifying_key.to_string()  # Add 0x04 prefix
            
            # MySocial address: BLAKE2b(0x02 || pub_key_bytes)
            flag_and_pubkey = bytes([0x02]) + pub_key_bytes  # 0x02 = secp256r1 flag
            address_bytes = hashlib.blake2b(flag_and_pubkey, digest_size=32).digest()
            self.address = "0x" + address_bytes.hex()
    
    def sign_transaction(self, message: bytes) -> str:
        """
        Sign a transaction message
        
        Args:
            message: Transaction data to sign
            
        Returns:
            Hex-encoded signature
        """
        if self.curve == "ed25519":
            return self._sign_ed25519(message)
        elif self.curve == "secp256k1":
            return self._sign_secp256k1(message)
        elif self.curve == "secp256r1":
            return self._sign_secp256r1(message)
        else:
            raise ValueError(f"Unsupported curve: {self.curve}")
    
    def _sign_ed25519(self, message: bytes) -> str:
        """Sign with Ed25519"""
        private_key_bytes = bytes.fromhex(self.private_key_hex.replace("0x", ""))
        signing_key = Ed25519SigningKey(private_key_bytes)
        
        # Hash message with BLAKE2b
        message_hash = hashlib.blake2b(message, digest_size=32).digest()
        
        # Sign
        signature = signing_key.sign(message_hash, encoder=RawEncoder)
        return signature.signature.hex()
    
    def _sign_secp256k1(self, message: bytes) -> str:
        """Sign with secp256k1 (Ethereum-compatible)"""
        account = Account.from_key(self.private_key_hex)
        
        # Hash message (MySocial transaction format)
        message_hash = hashlib.blake2b(message, digest_size=32).digest()
        
        # Sign
        signed_message = account.signHash(message_hash)
        
        # Return signature (r, s, v format)
        signature = signed_message.signature.hex()
        return signature
    
    def _sign_secp256r1(self, message: bytes) -> str:
        """Sign with secp256r1 (P-256)"""
        private_key_bytes = bytes.fromhex(self.private_key_hex.replace("0x", ""))
        signing_key = SigningKey.from_string(private_key_bytes, curve=NIST256p)
        
        # Hash message
        message_hash = hashlib.blake2b(message, digest_size=32).digest()
        
        # Sign
        signature = signing_key.sign_digest(message_hash, sigencode=sigencode_string)
        return signature.hex()
    
    def get_address(self) -> str:
        """Get wallet address"""
        return self.address
    
    def get_public_key(self) -> str:
        """Get public key in hex format"""
        if self.curve == "ed25519":
            private_key_bytes = bytes.fromhex(self.private_key_hex.replace("0x", ""))
            signing_key = Ed25519SigningKey(private_key_bytes)
            verify_key = signing_key.verify_key
            return bytes(verify_key).hex()
        
        elif self.curve == "secp256k1":
            account = Account.from_key(self.private_key_hex)
            # Get uncompressed public key
            return account._key_obj.public_key.to_hex()
        
        elif self.curve == "secp256r1":
            private_key_bytes = bytes.fromhex(self.private_key_hex.replace("0x", ""))
            signing_key = SigningKey.from_string(private_key_bytes, curve=NIST256p)
            verifying_key = signing_key.get_verifying_key()
            return verifying_key.to_string().hex()
    
    @staticmethod
    def generate_mnemonic() -> str:
        """Generate a new BIP39 mnemonic"""
        from bip_utils import Bip39MnemonicGenerator, Bip39WordsNum
        mnemonic = Bip39MnemonicGenerator().FromWordsNumber(Bip39WordsNum.WORDS_NUM_24)
        return str(mnemonic)


def load_oracle_wallet() -> MySocialWallet:
    """
    Load oracle wallet from environment variables
    
    Environment variables:
    - MYS_ORACLE_PRIVATE_KEY: Direct private key (Base64 or hex)
    - MYS_ORACLE_MNEMONIC: BIP39 mnemonic
    - MYS_ORACLE_CURVE: "ed25519", "secp256k1", or "secp256r1" (default: ed25519)
    """
    private_key = os.getenv("MYS_ORACLE_PRIVATE_KEY")
    mnemonic = os.getenv("MYS_ORACLE_MNEMONIC")
    curve = os.getenv("MYS_ORACLE_CURVE", "ed25519")  # Default to Ed25519
    
    if not private_key and not mnemonic:
        raise ValueError("Must set MYS_ORACLE_PRIVATE_KEY or MYS_ORACLE_MNEMONIC")
    
    if private_key:
        wallet = MySocialWallet(private_key=private_key, curve=curve)
    else:
        wallet = MySocialWallet(mnemonic=mnemonic, curve=curve)
    
    logger.info("Oracle wallet loaded", 
               address=wallet.get_address(), 
               curve=curve)
    
    return wallet

