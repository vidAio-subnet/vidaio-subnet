"""
Secret Provisioning Client

This module provides functionality for the miner to receive secrets from
the validator through Gramine's Secret Provisioning mechanism.

In the SGX enclave, the miner:
1. Generates an SGX Quote proving its identity
2. Sends the quote to the validator
3. Receives encrypted secrets (session key) after verification
4. Uses the session key to decrypt task payloads
"""

import os
import json
import socket
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger


# Gramine attestation pseudo-filesystem paths
ATTESTATION_USER_REPORT_DATA = "/dev/attestation/user_report_data"
ATTESTATION_QUOTE = "/dev/attestation/quote"
ATTESTATION_TYPE = "/dev/attestation/attestation_type"


@dataclass
class SecretProvisioningConfig:
    """Configuration for secret provisioning."""
    validator_host: str
    validator_port: int
    timeout_seconds: int = 30
    

class SecretProvisioningClient:
    """
    Client for receiving secrets from validators.
    
    This runs inside the SGX enclave and handles:
    1. Reading attestation quote from Gramine's pseudo-filesystem
    2. Sending quote to validator for verification
    3. Receiving and storing session keys
    """
    
    def __init__(self, config: Optional[SecretProvisioningConfig] = None):
        """
        Initialize secret provisioning client.
        
        Args:
            config: Configuration for connecting to validator
        """
        self.config = config
        self._is_inside_enclave = self._check_enclave_environment()
        self._session_key: Optional[bytes] = None
        self._session_key_id: Optional[str] = None
        
        if self._is_inside_enclave:
            logger.info("Running inside SGX enclave - attestation available")
        else:
            logger.warning("Not running inside SGX enclave - using mock attestation")
    
    def _check_enclave_environment(self) -> bool:
        """Check if we're running inside an SGX enclave."""
        return os.path.exists(ATTESTATION_QUOTE)
    
    def generate_quote(self, user_data: bytes = b"") -> bytes:
        """
        Generate an SGX Quote for attestation.
        
        The quote proves to the validator that we're running in a genuine
        SGX enclave with specific code (identified by MRENCLAVE).
        
        Args:
            user_data: Optional user data to embed in the quote (up to 64 bytes)
            
        Returns:
            Raw SGX quote bytes
        """
        if self._is_inside_enclave:
            return self._generate_quote_sgx(user_data)
        else:
            return self._generate_quote_mock(user_data)
    
    def _generate_quote_sgx(self, user_data: bytes) -> bytes:
        """
        Generate quote using Gramine's attestation pseudo-filesystem.
        
        Process:
        1. Write user_report_data to /dev/attestation/user_report_data
        2. Read quote from /dev/attestation/quote
        """
        # Pad or truncate user_data to 64 bytes
        user_data_padded = user_data[:64].ljust(64, b'\x00')
        
        try:
            # Write user report data
            with open(ATTESTATION_USER_REPORT_DATA, 'wb') as f:
                f.write(user_data_padded)
            
            # Read quote
            with open(ATTESTATION_QUOTE, 'rb') as f:
                quote = f.read()
            
            logger.info(f"Generated SGX quote: {len(quote)} bytes")
            return quote
            
        except IOError as e:
            logger.error(f"Failed to generate SGX quote: {e}")
            raise RuntimeError(f"SGX quote generation failed: {e}")
    
    def _generate_quote_mock(self, user_data: bytes) -> bytes:
        """
        Generate a mock quote for development/testing.
        
        WARNING: This should NOT be used in production!
        """
        from .attestation_verifier import create_mock_quote
        
        # Use a placeholder MRENCLAVE
        mock_mrenclave = "0" * 64
        
        logger.warning("Generated mock quote - NOT SECURE FOR PRODUCTION")
        return create_mock_quote(
            mrenclave=mock_mrenclave,
            is_debug=True
        )
    
    def get_attestation_type(self) -> str:
        """
        Get the attestation type supported by this environment.
        
        Returns:
            "dcap", "epid", or "none"
        """
        if not self._is_inside_enclave:
            return "none"
        
        try:
            with open(ATTESTATION_TYPE, 'r') as f:
                return f.read().strip()
        except IOError:
            return "none"
    
    def request_session_key(
        self,
        validator_host: str,
        validator_port: int,
        timeout: int = 30
    ) -> tuple[str, bytes]:
        """
        Request a session key from the validator.
        
        Process:
        1. Generate SGX quote
        2. Send quote to validator
        3. Validator verifies quote and returns encrypted session key
        4. Store session key for later use
        
        Args:
            validator_host: Hostname of the validator
            validator_port: Port for attestation service
            timeout: Connection timeout in seconds
            
        Returns:
            Tuple of (session_key_id, session_key)
            
        Raises:
            RuntimeError: If attestation fails
        """
        # Generate quote with random challenge
        import secrets
        challenge = secrets.token_bytes(32)
        quote = self.generate_quote(challenge)
        
        # Send to validator
        try:
            response = self._send_attestation_request(
                validator_host, validator_port, quote, challenge, timeout
            )
        except Exception as e:
            logger.error(f"Attestation request failed: {e}")
            raise RuntimeError(f"Failed to request session key: {e}")
        
        # Parse response
        if not response.get("success"):
            error = response.get("error", "Unknown error")
            logger.error(f"Attestation rejected: {error}")
            raise RuntimeError(f"Attestation rejected: {error}")
        
        # Extract session key
        import base64
        session_key_id = response["session_key_id"]
        session_key = base64.b64decode(response["encrypted_session_key"])
        
        # In full implementation, the session key would be encrypted with
        # the enclave's public key and decrypted here. For now, we receive
        # it directly (assumes secure channel).
        
        self._session_key_id = session_key_id
        self._session_key = session_key
        
        logger.info(f"Received session key: {session_key_id}")
        return session_key_id, session_key
    
    def _send_attestation_request(
        self,
        host: str,
        port: int,
        quote: bytes,
        challenge: bytes,
        timeout: int
    ) -> Dict[str, Any]:
        """
        Send attestation request to validator.
        
        Uses a simple JSON-over-TCP protocol. In production, this would
        be RA-TLS (Remote Attestation TLS).
        """
        import base64
        
        request = {
            "type": "attestation",
            "quote": base64.b64encode(quote).decode('utf-8'),
            "challenge": base64.b64encode(challenge).decode('utf-8'),
        }
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
            
            # Send request
            request_bytes = json.dumps(request).encode('utf-8')
            sock.sendall(len(request_bytes).to_bytes(4, 'big'))
            sock.sendall(request_bytes)
            
            # Receive response
            response_len = int.from_bytes(sock.recv(4), 'big')
            response_bytes = b""
            while len(response_bytes) < response_len:
                chunk = sock.recv(min(4096, response_len - len(response_bytes)))
                if not chunk:
                    raise RuntimeError("Connection closed unexpectedly")
                response_bytes += chunk
            
            return json.loads(response_bytes.decode('utf-8'))
    
    def get_session_key(self) -> Optional[tuple[str, bytes]]:
        """
        Get the current session key.
        
        Returns:
            Tuple of (session_key_id, session_key) or None if not yet obtained
        """
        if self._session_key_id and self._session_key:
            return self._session_key_id, self._session_key
        return None
    
    def clear_session_key(self) -> None:
        """
        Clear the current session key from memory.
        
        Call this when done processing to minimize key exposure.
        """
        self._session_key_id = None
        self._session_key = None
        logger.debug("Session key cleared from memory")


class SecretProvisioningServer:
    """
    Server for provisioning secrets to verified enclaves.
    
    This runs on the validator side and handles:
    1. Receiving attestation quotes from miners
    2. Verifying quotes using DCAP
    3. Provisioning session keys to verified enclaves
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8443,
        policy: Optional["AttestationPolicy"] = None
    ):
        """
        Initialize secret provisioning server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            policy: Attestation policy for verification
        """
        self.host = host
        self.port = port
        
        from .attestation_verifier import AttestationVerifier, DEFAULT_MINER_POLICY
        self.policy = policy or DEFAULT_MINER_POLICY
        self.verifier = AttestationVerifier(self.policy)
        
        # Store session keys for lookup
        self._session_keys: Dict[str, bytes] = {}
    
    def handle_attestation(self, quote: bytes, challenge: bytes) -> Dict[str, Any]:
        """
        Handle an attestation request.
        
        Args:
            quote: SGX quote from miner
            challenge: Random challenge that should be embedded in quote
            
        Returns:
            Response dictionary with session key or error
        """
        from .attestation_verifier import AttestationResult
        from .tee_crypto import TEECrypto
        
        # Verify quote
        result, report = self.verifier.verify_quote(quote)
        
        if result != AttestationResult.SUCCESS:
            logger.warning(f"Attestation failed: {result.value}")
            return {
                "success": False,
                "error": result.value,
            }
        
        # Generate session key
        crypto = TEECrypto()
        session_key_id, session_key = crypto.generate_session_key()
        
        # Store for later use
        self._session_keys[session_key_id] = session_key
        
        # Return session key
        # In full implementation, this would be encrypted with enclave's public key
        import base64
        return {
            "success": True,
            "session_key_id": session_key_id,
            "encrypted_session_key": base64.b64encode(session_key).decode('utf-8'),
            "enclave_report": report.to_dict() if report else None,
        }
    
    def get_session_key(self, session_key_id: str) -> Optional[bytes]:
        """Get a session key by ID."""
        return self._session_keys.get(session_key_id)
    
    def remove_session_key(self, session_key_id: str) -> None:
        """Remove a session key (after task completion)."""
        self._session_keys.pop(session_key_id, None)
