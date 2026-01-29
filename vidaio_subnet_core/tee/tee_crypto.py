"""
TEE Cryptographic Utilities

This module provides cryptographic functions for secure communication between
validators and miners running inside Intel SGX enclaves. All encryption uses
AES-256-GCM for authenticated encryption.

The encryption scheme ensures:
- Video URLs are encrypted and only decryptable inside the SGX enclave
- Storage credentials are protected end-to-end
- Result paths are hidden from miners
"""

import os
import json
import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
import base64
from loguru import logger


# Constants
KEY_SIZE = 32  # 256-bit keys
NONCE_SIZE = 12  # 96-bit nonce for AES-GCM
SESSION_KEY_ID_SIZE = 16  # 128-bit session key identifier


@dataclass
class EncryptedPayload:
    """
    Container for encrypted task payload.
    
    The encrypted_blob contains AES-GCM encrypted JSON with:
    - reference_video_url: URL to download the input video
    - storage_credentials: S3/MinIO credentials for result upload
    - task_params: Task-specific parameters (upscaling/compression settings)
    - result_object_key: Path where result should be uploaded
    """
    encrypted_blob: bytes
    nonce: bytes
    session_key_id: str
    
    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary with base64-encoded binary fields."""
        return {
            "encrypted_blob": base64.b64encode(self.encrypted_blob).decode('utf-8'),
            "nonce": base64.b64encode(self.nonce).decode('utf-8'),
            "session_key_id": self.session_key_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EncryptedPayload":
        """Deserialize from dictionary with base64-encoded binary fields."""
        return cls(
            encrypted_blob=base64.b64decode(data["encrypted_blob"]),
            nonce=base64.b64decode(data["nonce"]),
            session_key_id=data["session_key_id"],
        )


@dataclass 
class StorageCredentials:
    """
    Credentials for uploading results to validator's storage.
    These are provisioned encrypted and only decrypted inside the enclave.
    """
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str
    region: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_url": self.endpoint_url,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
            "bucket": self.bucket,
            "region": self.region,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageCredentials":
        return cls(
            endpoint_url=data["endpoint_url"],
            access_key=data["access_key"],
            secret_key=data["secret_key"],
            bucket=data["bucket"],
            region=data.get("region"),
        )


@dataclass
class TaskPayload:
    """
    Plaintext task payload structure.
    This is the data that gets encrypted before sending to miners.
    """
    reference_video_url: str
    storage_credentials: StorageCredentials
    result_object_key: str
    task_type: str  # "upscaling" or "compression"
    task_params: Dict[str, Any]  # Task-specific parameters
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_video_url": self.reference_video_url,
            "storage_credentials": self.storage_credentials.to_dict(),
            "result_object_key": self.result_object_key,
            "task_type": self.task_type,
            "task_params": self.task_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPayload":
        return cls(
            reference_video_url=data["reference_video_url"],
            storage_credentials=StorageCredentials.from_dict(data["storage_credentials"]),
            result_object_key=data["result_object_key"],
            task_type=data["task_type"],
            task_params=data["task_params"],
        )


class TEECrypto:
    """
    Cryptographic utilities for TEE communication.
    
    Provides methods for:
    - Generating session keys for validator-miner communication
    - Encrypting task payloads (URLs, credentials, parameters)
    - Decrypting payloads inside the enclave
    - Deriving keys from shared secrets (for attestation-based key exchange)
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize TEE crypto utilities.
        
        Args:
            master_key: Optional master key for deterministic key derivation.
                       If not provided, keys will be randomly generated.
        """
        self._master_key = master_key
        self._session_keys: Dict[str, bytes] = {}
    
    def generate_session_key(self) -> tuple[str, bytes]:
        """
        Generate a new session key for encrypting communications.
        
        Returns:
            Tuple of (session_key_id, session_key)
        """
        session_key_id = secrets.token_hex(SESSION_KEY_ID_SIZE)
        session_key = secrets.token_bytes(KEY_SIZE)
        
        # Store for later retrieval
        self._session_keys[session_key_id] = session_key
        
        logger.debug(f"Generated session key with ID: {session_key_id}")
        return session_key_id, session_key
    
    def get_session_key(self, session_key_id: str) -> Optional[bytes]:
        """
        Retrieve a previously generated session key.
        
        Args:
            session_key_id: The session key identifier
            
        Returns:
            The session key bytes, or None if not found
        """
        return self._session_keys.get(session_key_id)
    
    def store_session_key(self, session_key_id: str, session_key: bytes) -> None:
        """
        Store a session key received from attestation.
        
        Args:
            session_key_id: The session key identifier
            session_key: The session key bytes
        """
        self._session_keys[session_key_id] = session_key
        logger.debug(f"Stored session key with ID: {session_key_id}")
    
    @staticmethod
    def derive_key_from_shared_secret(
        shared_secret: bytes,
        salt: Optional[bytes] = None,
        info: bytes = b"tee-session-key"
    ) -> bytes:
        """
        Derive a session key from a shared secret using HKDF.
        
        This is used after attestation to derive keys from the
        attestation-established shared secret.
        
        Args:
            shared_secret: The shared secret from attestation
            salt: Optional salt for HKDF
            info: Context info for key derivation
            
        Returns:
            Derived key bytes
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(shared_secret)
    
    @staticmethod
    def encrypt_payload(
        payload: TaskPayload,
        session_key: bytes,
        session_key_id: str
    ) -> EncryptedPayload:
        """
        Encrypt a task payload for transmission to the miner enclave.
        
        Uses AES-256-GCM for authenticated encryption.
        
        Args:
            payload: The plaintext task payload
            session_key: The encryption key
            session_key_id: Identifier for the session key
            
        Returns:
            EncryptedPayload containing the encrypted data
        """
        # Generate random nonce
        nonce = secrets.token_bytes(NONCE_SIZE)
        
        # Serialize payload to JSON
        plaintext = json.dumps(payload.to_dict()).encode('utf-8')
        
        # Encrypt with AES-GCM
        aesgcm = AESGCM(session_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        logger.debug(f"Encrypted payload: {len(plaintext)} bytes -> {len(ciphertext)} bytes")
        
        return EncryptedPayload(
            encrypted_blob=ciphertext,
            nonce=nonce,
            session_key_id=session_key_id
        )
    
    def decrypt_payload(
        self,
        encrypted: EncryptedPayload,
        session_key: Optional[bytes] = None
    ) -> TaskPayload:
        """
        Decrypt a task payload inside the enclave.
        
        Args:
            encrypted: The encrypted payload
            session_key: The decryption key. If not provided, will look up
                        by session_key_id.
                        
        Returns:
            Decrypted TaskPayload
            
        Raises:
            ValueError: If decryption fails or session key not found
        """
        # Get session key if not provided
        if session_key is None:
            session_key = self.get_session_key(encrypted.session_key_id)
            if session_key is None:
                raise ValueError(f"Session key not found: {encrypted.session_key_id}")
        
        # Decrypt with AES-GCM
        aesgcm = AESGCM(session_key)
        try:
            plaintext = aesgcm.decrypt(
                encrypted.nonce,
                encrypted.encrypted_blob,
                None
            )
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt payload - invalid key or corrupted data")
        
        # Parse JSON and create TaskPayload
        data = json.loads(plaintext.decode('utf-8'))
        return TaskPayload.from_dict(data)
    
    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """
        Compute SHA-256 checksum of data.
        
        Used for result verification.
        
        Args:
            data: The data to checksum
            
        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def compute_file_checksum(filepath: str) -> str:
        """
        Compute SHA-256 checksum of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Hex-encoded SHA-256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


# Convenience functions for common operations

def encrypt_task_for_miner(
    video_url: str,
    storage_credentials: StorageCredentials,
    result_object_key: str,
    task_type: str,
    task_params: Dict[str, Any],
    session_key: bytes,
    session_key_id: str
) -> EncryptedPayload:
    """
    Convenience function to encrypt a complete task for a miner.
    
    Args:
        video_url: URL to the input video
        storage_credentials: Credentials for result upload
        result_object_key: Path where result should be uploaded
        task_type: "upscaling" or "compression"
        task_params: Task-specific parameters
        session_key: Encryption key from attestation
        session_key_id: Session key identifier
        
    Returns:
        EncryptedPayload ready for transmission
    """
    payload = TaskPayload(
        reference_video_url=video_url,
        storage_credentials=storage_credentials,
        result_object_key=result_object_key,
        task_type=task_type,
        task_params=task_params
    )
    
    return TEECrypto.encrypt_payload(payload, session_key, session_key_id)
