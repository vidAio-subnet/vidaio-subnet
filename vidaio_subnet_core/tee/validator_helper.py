"""
Validator TEE Integration Helper

This module provides helper functions for validators to:
1. Perform attestation with miners
2. Create encrypted task payloads
3. Retrieve results from storage
4. Verify result integrity

The validator never exposes video URLs to miners - all sensitive data
is encrypted and only decryptable inside the miner's SGX enclave.
"""

import asyncio
import secrets
import uuid
import base64
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

import bittensor as bt

from vidaio_subnet_core.protocol import (
    TEEAttestationProtocol,
    TEEVideoUpscalingProtocol,
    TEEVideoCompressionProtocol,
    EncryptedTaskPayload,
    TEEMinerResponse,
    TEECapabilities,
    Version,
)
from vidaio_subnet_core.tee.tee_crypto import (
    TEECrypto,
    EncryptedPayload,
    TaskPayload,
    StorageCredentials,
    encrypt_task_for_miner,
)
from vidaio_subnet_core.tee.attestation_verifier import (
    AttestationVerifier,
    AttestationPolicy,
    AttestationResult,
    EnclaveReport,
)
from vidaio_subnet_core.tee.secret_provisioning import (
    SecretProvisioningServer,
)


@dataclass
class MinerSession:
    """Represents an authenticated session with a miner."""
    miner_uid: int
    hotkey: str
    session_key_id: str
    session_key: bytes
    enclave_report: Optional[EnclaveReport] = None
    attestation_time: float = 0.0
    is_valid: bool = False


@dataclass
class TaskResult:
    """Result of a TEE video processing task."""
    success: bool
    miner_uid: int
    processing_time_seconds: float
    result_checksum: str
    result_object_key: str
    error_message: Optional[str] = None


class ValidatorTEEHelper:
    """
    Helper class for validators to interact with TEE-enabled miners.
    
    Handles:
    - Attestation and session key provisioning
    - Encrypted payload creation
    - Result retrieval and verification
    """
    
    def __init__(
        self,
        storage_credentials: StorageCredentials,
        attestation_policy: Optional[AttestationPolicy] = None,
    ):
        """
        Initialize validator TEE helper.
        
        Args:
            storage_credentials: Credentials for the validator's result storage
            attestation_policy: Policy for verifying miner enclaves
        """
        self.storage_credentials = storage_credentials
        self.crypto = TEECrypto()
        self.secret_server = SecretProvisioningServer(policy=attestation_policy)
        self.verifier = self.secret_server.verifier
        
        # Track miner sessions
        self.miner_sessions: Dict[int, MinerSession] = {}  # miner_uid -> MinerSession
        
        logger.info("Validator TEE Helper initialized")
    
    # =========================================================================
    # Attestation
    # =========================================================================
    
    async def attest_miner(
        self,
        dendrite: bt.dendrite,
        miner_axon: bt.axon,
        miner_uid: int,
        timeout: float = 30.0,
    ) -> Optional[MinerSession]:
        """
        Perform attestation with a miner to establish a secure session.
        
        Args:
            dendrite: Bittensor dendrite for communication
            miner_axon: Axon info for the miner
            miner_uid: UID of the miner
            timeout: Timeout for attestation request
            
        Returns:
            MinerSession if attestation successful, None otherwise
        """
        logger.info(f"Starting attestation with miner {miner_uid}")
        
        # Generate challenge
        challenge = secrets.token_bytes(32)
        
        # Create attestation synapse
        synapse = TEEAttestationProtocol(
            challenge=base64.b64encode(challenge).decode('utf-8'),
        )
        
        # Send to miner
        try:
            responses = await dendrite.forward(
                axons=[miner_axon],
                synapse=synapse,
                timeout=timeout,
            )
            response = responses[0] if responses else None
        except Exception as e:
            logger.error(f"Attestation request failed for miner {miner_uid}: {e}")
            return None
        
        if not response or not response.sgx_quote:
            logger.warning(f"Miner {miner_uid} did not return SGX quote")
            return None
        
        # Verify quote
        quote = base64.b64decode(response.sgx_quote)
        result = self.secret_server.handle_attestation(quote, challenge)
        
        if not result.get("success"):
            logger.warning(f"Attestation failed for miner {miner_uid}: {result.get('error')}")
            return None
        
        # Create session
        session = MinerSession(
            miner_uid=miner_uid,
            hotkey=miner_axon.hotkey,
            session_key_id=result["session_key_id"],
            session_key=base64.b64decode(result["encrypted_session_key"]),
            enclave_report=EnclaveReport.from_dict(result["enclave_report"]) if result.get("enclave_report") else None,
            attestation_time=asyncio.get_event_loop().time(),
            is_valid=True,
        )
        
        # Store session key in crypto for later use
        self.crypto.store_session_key(session.session_key_id, session.session_key)
        
        # Store session
        self.miner_sessions[miner_uid] = session
        
        logger.info(f"Attestation successful for miner {miner_uid}")
        logger.info(f"  Session key ID: {session.session_key_id}")
        if session.enclave_report:
            logger.info(f"  MRENCLAVE: {session.enclave_report.mrenclave_hex()[:16]}...")
        
        # Send session key back to miner
        await self._provision_session_key(dendrite, miner_axon, session)
        
        return session
    
    async def _provision_session_key(
        self,
        dendrite: bt.dendrite,
        miner_axon: bt.axon,
        session: MinerSession,
    ) -> bool:
        """
        Provision session key to miner after successful attestation.
        """
        synapse = TEEAttestationProtocol(
            session_key_id=session.session_key_id,
            encrypted_session_key=base64.b64encode(session.session_key).decode('utf-8'),
            attestation_success=True,
        )
        
        try:
            await dendrite.forward(
                axons=[miner_axon],
                synapse=synapse,
                timeout=10.0,
            )
            logger.info(f"Session key provisioned to miner {session.miner_uid}")
            return True
        except Exception as e:
            logger.error(f"Failed to provision session key: {e}")
            return False
    
    def get_miner_session(self, miner_uid: int) -> Optional[MinerSession]:
        """Get existing session for a miner."""
        return self.miner_sessions.get(miner_uid)
    
    def is_miner_attested(self, miner_uid: int) -> bool:
        """Check if miner has a valid attestation."""
        session = self.miner_sessions.get(miner_uid)
        return session is not None and session.is_valid
    
    # =========================================================================
    # Task Creation
    # =========================================================================
    
    def create_upscaling_task(
        self,
        video_url: str,
        task_type: str = "HD24K",
        miner_uid: int = 0,
    ) -> Tuple[Optional[TEEVideoUpscalingProtocol], Optional[str]]:
        """
        Create an encrypted upscaling task for a miner.
        
        Args:
            video_url: URL of the video to upscale (will be encrypted)
            task_type: Upscaling type ("HD24K", "SD2HD", etc.)
            miner_uid: UID of the target miner
            
        Returns:
            Tuple of (protocol synapse, result_object_key) or (None, None) if failed
        """
        session = self.miner_sessions.get(miner_uid)
        if not session or not session.is_valid:
            logger.error(f"No valid session for miner {miner_uid} - attestation required")
            return None, None
        
        # Generate unique result path
        result_object_key = f"results/upscaling/{uuid.uuid4()}.mp4"
        round_id = str(uuid.uuid4())
        
        # Create encrypted payload
        encrypted = encrypt_task_for_miner(
            video_url=video_url,
            storage_credentials=self.storage_credentials,
            result_object_key=result_object_key,
            task_type="upscaling",
            task_params={
                "task_type": task_type,
            },
            session_key=session.session_key,
            session_key_id=session.session_key_id,
        )
        
        # Create synapse
        synapse = TEEVideoUpscalingProtocol(
            round_id=round_id,
            encrypted_payload=EncryptedTaskPayload(
                encrypted_blob=base64.b64encode(encrypted.encrypted_blob).decode('utf-8'),
                nonce=base64.b64encode(encrypted.nonce).decode('utf-8'),
                session_key_id=encrypted.session_key_id,
            ),
        )
        
        logger.info(f"Created upscaling task for miner {miner_uid}")
        logger.debug(f"  Result path: {result_object_key}")
        
        return synapse, result_object_key
    
    def create_compression_task(
        self,
        video_url: str,
        vmaf_threshold: float = 90.0,
        target_codec: str = "av1",
        codec_mode: str = "CRF",
        target_bitrate: float = 5.0,
        miner_uid: int = 0,
    ) -> Tuple[Optional[TEEVideoCompressionProtocol], Optional[str]]:
        """
        Create an encrypted compression task for a miner.
        
        Args:
            video_url: URL of the video to compress (will be encrypted)
            vmaf_threshold: Target VMAF score
            target_codec: Codec to use
            codec_mode: Encoding mode
            target_bitrate: Target bitrate in Mbps
            miner_uid: UID of the target miner
            
        Returns:
            Tuple of (protocol synapse, result_object_key) or (None, None) if failed
        """
        session = self.miner_sessions.get(miner_uid)
        if not session or not session.is_valid:
            logger.error(f"No valid session for miner {miner_uid} - attestation required")
            return None, None
        
        # Generate unique result path
        result_object_key = f"results/compression/{uuid.uuid4()}.mp4"
        round_id = str(uuid.uuid4())
        
        # Create encrypted payload
        encrypted = encrypt_task_for_miner(
            video_url=video_url,
            storage_credentials=self.storage_credentials,
            result_object_key=result_object_key,
            task_type="compression",
            task_params={
                "vmaf_threshold": vmaf_threshold,
                "target_codec": target_codec,
                "codec_mode": codec_mode,
                "target_bitrate": target_bitrate,
            },
            session_key=session.session_key,
            session_key_id=session.session_key_id,
        )
        
        # Create synapse
        synapse = TEEVideoCompressionProtocol(
            round_id=round_id,
            encrypted_payload=EncryptedTaskPayload(
                encrypted_blob=base64.b64encode(encrypted.encrypted_blob).decode('utf-8'),
                nonce=base64.b64encode(encrypted.nonce).decode('utf-8'),
                session_key_id=encrypted.session_key_id,
            ),
        )
        
        logger.info(f"Created compression task for miner {miner_uid}")
        logger.debug(f"  Result path: {result_object_key}")
        
        return synapse, result_object_key
    
    # =========================================================================
    # Result Handling
    # =========================================================================
    
    def process_response(
        self,
        response: TEEMinerResponse,
        result_object_key: str,
        miner_uid: int,
    ) -> TaskResult:
        """
        Process miner response and create TaskResult.
        
        The response contains only status and checksum - no URLs.
        The validator retrieves the result using the known object key.
        """
        return TaskResult(
            success=response.success,
            miner_uid=miner_uid,
            processing_time_seconds=response.processing_time_seconds,
            result_checksum=response.result_checksum,
            result_object_key=result_object_key,
            error_message=response.error_message,
        )
    
    async def retrieve_result(
        self,
        result_object_key: str,
        local_path: str,
    ) -> bool:
        """
        Retrieve processed video from storage.
        
        The validator retrieves results directly - miners never see this URL.
        
        Args:
            result_object_key: Object key in storage
            local_path: Path to save downloaded file
            
        Returns:
            True if successful
        """
        try:
            import boto3
            from botocore.config import Config
            
            s3 = boto3.client(
                's3',
                endpoint_url=self.storage_credentials.endpoint_url,
                aws_access_key_id=self.storage_credentials.access_key,
                aws_secret_access_key=self.storage_credentials.secret_key,
                region_name=self.storage_credentials.region or 'us-east-1',
                config=Config(signature_version='s3v4')
            )
            
            s3.download_file(
                self.storage_credentials.bucket,
                result_object_key,
                local_path
            )
            
            logger.info(f"Retrieved result: {result_object_key} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retrieve result: {e}")
            return False
    
    def verify_result_checksum(
        self,
        local_path: str,
        expected_checksum: str,
    ) -> bool:
        """
        Verify that downloaded result matches expected checksum.
        
        Args:
            local_path: Path to downloaded file
            expected_checksum: Expected SHA-256 checksum from miner
            
        Returns:
            True if checksum matches
        """
        actual_checksum = TEECrypto.compute_file_checksum(local_path)
        matches = actual_checksum == expected_checksum
        
        if matches:
            logger.info(f"Checksum verified: {expected_checksum[:16]}...")
        else:
            logger.warning(f"Checksum mismatch! Expected: {expected_checksum}, Actual: {actual_checksum}")
        
        return matches
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def invalidate_session(self, miner_uid: int) -> None:
        """Invalidate a miner's session (e.g., after task failure)."""
        if miner_uid in self.miner_sessions:
            session = self.miner_sessions[miner_uid]
            session.is_valid = False
            # Remove session key
            self.secret_server.remove_session_key(session.session_key_id)
            logger.info(f"Invalidated session for miner {miner_uid}")
    
    def cleanup_expired_sessions(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up expired sessions.
        
        Args:
            max_age_seconds: Maximum session age in seconds
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = asyncio.get_event_loop().time()
        expired = []
        
        for miner_uid, session in self.miner_sessions.items():
            if current_time - session.attestation_time > max_age_seconds:
                expired.append(miner_uid)
        
        for miner_uid in expired:
            self.invalidate_session(miner_uid)
            del self.miner_sessions[miner_uid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)


def create_storage_credentials_from_env() -> StorageCredentials:
    """
    Create storage credentials from environment variables.
    
    Expected env vars:
    - MINIO_ENDPOINT or S3_ENDPOINT
    - MINIO_ACCESS_KEY or AWS_ACCESS_KEY_ID
    - MINIO_SECRET_KEY or AWS_SECRET_ACCESS_KEY
    - MINIO_BUCKET or S3_BUCKET
    """
    import os
    
    return StorageCredentials(
        endpoint_url=os.environ.get("MINIO_ENDPOINT", os.environ.get("S3_ENDPOINT", "http://localhost:9000")),
        access_key=os.environ.get("MINIO_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", "")),
        secret_key=os.environ.get("MINIO_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "")),
        bucket=os.environ.get("MINIO_BUCKET", os.environ.get("S3_BUCKET", "vidaio-results")),
        region=os.environ.get("AWS_REGION", "us-east-1"),
    )
