# vidaio_subnet_core/tee - Trusted Execution Environment Module
#
# This module provides TEE (Intel SGX) integration for the vidaio-subnet,
# enabling secure video processing where miners cannot access video URLs
# or content.

from .tee_crypto import TEECrypto, EncryptedPayload, TaskPayload, StorageCredentials
from .attestation_verifier import AttestationVerifier, EnclaveReport, AttestationPolicy, AttestationResult
from .secret_provisioning import SecretProvisioningClient, SecretProvisioningServer
from .validator_helper import ValidatorTEEHelper, MinerSession, TaskResult, create_storage_credentials_from_env

__all__ = [
    # Crypto
    "TEECrypto",
    "EncryptedPayload",
    "TaskPayload",
    "StorageCredentials",
    # Attestation
    "AttestationVerifier",
    "EnclaveReport",
    "AttestationPolicy",
    "AttestationResult",
    # Secret Provisioning
    "SecretProvisioningClient",
    "SecretProvisioningServer",
    # Validator Helper
    "ValidatorTEEHelper",
    "MinerSession",
    "TaskResult",
    "create_storage_credentials_from_env",
]
