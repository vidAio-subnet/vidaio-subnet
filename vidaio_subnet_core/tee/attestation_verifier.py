"""
SGX Attestation Verification

This module provides functionality for verifying Intel SGX attestation quotes
using DCAP (Data Center Attestation Primitives). Validators use this to verify
that miners are running inside genuine SGX enclaves before provisioning secrets.

The verification process:
1. Miner generates SGX Quote containing enclave measurements
2. Validator verifies the quote using DCAP libraries
3. Validator checks MRENCLAVE matches expected value
4. If verified, validator provisions encrypted secrets to the enclave
"""

import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
from loguru import logger
import base64


class AttestationResult(Enum):
    """Result of attestation verification."""
    SUCCESS = "success"
    INVALID_QUOTE = "invalid_quote"
    MRENCLAVE_MISMATCH = "mrenclave_mismatch"
    MRSIGNER_MISMATCH = "mrsigner_mismatch"
    DEBUG_ENCLAVE = "debug_enclave"
    QUOTE_EXPIRED = "quote_expired"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class EnclaveReport:
    """
    Enclave identity extracted from SGX Quote.
    
    Contains the measurements that uniquely identify an enclave:
    - MRENCLAVE: Hash of enclave code and data at build time
    - MRSIGNER: Hash of enclave signing key
    - ISV Product ID: Vendor-defined product identifier
    - ISV SVN: Security version number
    """
    mrenclave: bytes  # 32 bytes - hash of enclave code
    mrsigner: bytes   # 32 bytes - hash of signing key
    isv_prod_id: int  # Product ID (16-bit)
    isv_svn: int      # Security version number (16-bit)
    attributes: bytes # Enclave attributes (16 bytes)
    is_debug: bool    # Whether this is a debug enclave
    
    def mrenclave_hex(self) -> str:
        """Return MRENCLAVE as hex string."""
        return self.mrenclave.hex()
    
    def mrsigner_hex(self) -> str:
        """Return MRSIGNER as hex string."""
        return self.mrsigner.hex()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mrenclave": self.mrenclave.hex(),
            "mrsigner": self.mrsigner.hex(),
            "isv_prod_id": self.isv_prod_id,
            "isv_svn": self.isv_svn,
            "attributes": self.attributes.hex(),
            "is_debug": self.is_debug,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnclaveReport":
        """Deserialize from dictionary."""
        return cls(
            mrenclave=bytes.fromhex(data["mrenclave"]),
            mrsigner=bytes.fromhex(data["mrsigner"]),
            isv_prod_id=data["isv_prod_id"],
            isv_svn=data["isv_svn"],
            attributes=bytes.fromhex(data["attributes"]),
            is_debug=data["is_debug"],
        )


@dataclass
class AttestationPolicy:
    """
    Policy for attestation verification.
    
    Defines what enclave measurements are acceptable.
    """
    # List of acceptable MRENCLAVE values (hex strings)
    allowed_mrenclaves: List[str]
    
    # Optional: List of acceptable MRSIGNER values
    allowed_mrsigners: Optional[List[str]] = None
    
    # Minimum ISV SVN required
    min_isv_svn: int = 0
    
    # Whether to allow debug enclaves (should be False in production)
    allow_debug: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed_mrenclaves": self.allowed_mrenclaves,
            "allowed_mrsigners": self.allowed_mrsigners,
            "min_isv_svn": self.min_isv_svn,
            "allow_debug": self.allow_debug,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttestationPolicy":
        """Deserialize from dictionary."""
        return cls(
            allowed_mrenclaves=data["allowed_mrenclaves"],
            allowed_mrsigners=data.get("allowed_mrsigners"),
            min_isv_svn=data.get("min_isv_svn", 0),
            allow_debug=data.get("allow_debug", False),
        )


class AttestationVerifier:
    """
    Verifies SGX attestation quotes using DCAP.
    
    This class is used by validators to verify that miners are running
    inside genuine Intel SGX enclaves with the expected code.
    """
    
    def __init__(self, policy: AttestationPolicy):
        """
        Initialize attestation verifier with a policy.
        
        Args:
            policy: The attestation policy defining acceptable enclaves
        """
        self.policy = policy
        self._dcap_available = self._check_dcap_available()
        
        if not self._dcap_available:
            logger.warning("DCAP libraries not available - using mock verification")
    
    def _check_dcap_available(self) -> bool:
        """Check if DCAP verification libraries are available."""
        try:
            # Try to import DCAP verification library
            # In production, this would be the actual DCAP library
            # For now, we'll use a mock implementation
            return False
        except ImportError:
            return False
    
    def verify_quote(self, quote: bytes, user_data: Optional[bytes] = None) -> tuple[AttestationResult, Optional[EnclaveReport]]:
        """
        Verify an SGX quote.
        
        Args:
            quote: The raw SGX quote bytes
            user_data: Optional user data that should match the quote's report data
            
        Returns:
            Tuple of (AttestationResult, EnclaveReport or None)
        """
        if self._dcap_available:
            return self._verify_quote_dcap(quote, user_data)
        else:
            return self._verify_quote_mock(quote, user_data)
    
    def _verify_quote_dcap(self, quote: bytes, user_data: Optional[bytes]) -> tuple[AttestationResult, Optional[EnclaveReport]]:
        """
        Verify quote using DCAP libraries.
        
        This would use the actual Intel DCAP libraries in production.
        """
        # TODO: Implement actual DCAP verification
        # This requires:
        # 1. Intel SGX DCAP Quote Verification Library
        # 2. Access to Intel PCS or a local caching service
        raise NotImplementedError("DCAP verification not yet implemented")
    
    def _verify_quote_mock(self, quote: bytes, user_data: Optional[bytes]) -> tuple[AttestationResult, Optional[EnclaveReport]]:
        """
        Mock quote verification for development/testing.
        
        WARNING: This should NOT be used in production!
        """
        logger.warning("Using mock attestation verification - NOT SECURE FOR PRODUCTION")
        
        # Parse the mock quote format
        # Real quotes have a specific structure defined by Intel
        try:
            report = self._parse_mock_quote(quote)
        except Exception as e:
            logger.error(f"Failed to parse quote: {e}")
            return AttestationResult.INVALID_QUOTE, None
        
        # Check against policy
        result = self._check_policy(report)
        if result != AttestationResult.SUCCESS:
            return result, report
        
        return AttestationResult.SUCCESS, report
    
    def _parse_mock_quote(self, quote: bytes) -> EnclaveReport:
        """
        Parse a mock quote for testing.
        
        Real SGX quotes have a complex structure:
        - Header (48 bytes)
        - ISV Enclave Report (384 bytes)
        - Signature (variable)
        
        For mock purposes, we use a simplified JSON format.
        """
        import json
        
        try:
            # Try parsing as JSON (mock format)
            data = json.loads(quote.decode('utf-8'))
            return EnclaveReport.from_dict(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Try parsing as binary format
            if len(quote) < 64:
                raise ValueError("Quote too short")
            
            # Extract fields from binary (simplified mock)
            return EnclaveReport(
                mrenclave=quote[0:32],
                mrsigner=quote[32:64],
                isv_prod_id=int.from_bytes(quote[64:66], 'little') if len(quote) > 66 else 0,
                isv_svn=int.from_bytes(quote[66:68], 'little') if len(quote) > 68 else 0,
                attributes=quote[68:84] if len(quote) > 84 else bytes(16),
                is_debug=bool(quote[84]) if len(quote) > 84 else False,
            )
    
    def _check_policy(self, report: EnclaveReport) -> AttestationResult:
        """
        Check if an enclave report matches the policy.
        
        Args:
            report: The enclave report to check
            
        Returns:
            AttestationResult indicating success or failure reason
        """
        # Check debug mode
        if report.is_debug and not self.policy.allow_debug:
            logger.warning(f"Rejecting debug enclave: {report.mrenclave_hex()}")
            return AttestationResult.DEBUG_ENCLAVE
        
        # Check MRENCLAVE
        mrenclave_hex = report.mrenclave_hex()
        if mrenclave_hex not in self.policy.allowed_mrenclaves:
            logger.warning(f"MRENCLAVE mismatch: {mrenclave_hex} not in allowed list")
            return AttestationResult.MRENCLAVE_MISMATCH
        
        # Check MRSIGNER if policy specifies it
        if self.policy.allowed_mrsigners:
            mrsigner_hex = report.mrsigner_hex()
            if mrsigner_hex not in self.policy.allowed_mrsigners:
                logger.warning(f"MRSIGNER mismatch: {mrsigner_hex} not in allowed list")
                return AttestationResult.MRSIGNER_MISMATCH
        
        # Check minimum SVN
        if report.isv_svn < self.policy.min_isv_svn:
            logger.warning(f"ISV SVN too low: {report.isv_svn} < {self.policy.min_isv_svn}")
            return AttestationResult.VERIFICATION_FAILED
        
        logger.info(f"Attestation verified for MRENCLAVE: {mrenclave_hex[:16]}...")
        return AttestationResult.SUCCESS
    
    def extract_enclave_report(self, quote: bytes) -> Optional[EnclaveReport]:
        """
        Extract enclave report from quote without full verification.
        
        Useful for logging/debugging.
        
        Args:
            quote: The raw SGX quote bytes
            
        Returns:
            EnclaveReport or None if parsing fails
        """
        try:
            return self._parse_mock_quote(quote)
        except Exception as e:
            logger.error(f"Failed to extract enclave report: {e}")
            return None


def create_mock_quote(
    mrenclave: str,
    mrsigner: str = "0" * 64,
    isv_prod_id: int = 1,
    isv_svn: int = 1,
    is_debug: bool = False
) -> bytes:
    """
    Create a mock SGX quote for testing.
    
    Args:
        mrenclave: Hex-encoded MRENCLAVE (64 chars)
        mrsigner: Hex-encoded MRSIGNER (64 chars)
        isv_prod_id: Product ID
        isv_svn: Security version number
        is_debug: Whether this is a debug enclave
        
    Returns:
        Mock quote bytes
    """
    import json
    
    data = {
        "mrenclave": mrenclave,
        "mrsigner": mrsigner,
        "isv_prod_id": isv_prod_id,
        "isv_svn": isv_svn,
        "attributes": "0" * 32,
        "is_debug": is_debug,
    }
    
    return json.dumps(data).encode('utf-8')


# Default policy for vidaio-subnet miners
# This should be updated with actual MRENCLAVE values after building the enclave
DEFAULT_MINER_POLICY = AttestationPolicy(
    allowed_mrenclaves=[
        # Placeholder - replace with actual MRENCLAVE after building miner enclave
        "0" * 64,
    ],
    allow_debug=True,  # Set to False in production
    min_isv_svn=1,
)
