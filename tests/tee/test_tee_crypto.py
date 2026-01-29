"""
Tests for TEE Cryptographic Utilities

Tests the encryption/decryption, session key management, and
payload handling for TEE communication.
"""

import pytest
import secrets
import base64

from vidaio_subnet_core.tee.tee_crypto import (
    TEECrypto,
    EncryptedPayload,
    TaskPayload,
    StorageCredentials,
    encrypt_task_for_miner,
    KEY_SIZE,
    NONCE_SIZE,
)


class TestTEECrypto:
    """Tests for TEECrypto class."""
    
    def test_generate_session_key(self):
        """Test session key generation."""
        crypto = TEECrypto()
        session_key_id, session_key = crypto.generate_session_key()
        
        assert session_key_id is not None
        assert len(session_key) == KEY_SIZE
        assert isinstance(session_key, bytes)
    
    def test_store_and_retrieve_session_key(self):
        """Test storing and retrieving session keys."""
        crypto = TEECrypto()
        
        # Generate and store
        session_key_id, session_key = crypto.generate_session_key()
        
        # Retrieve
        retrieved = crypto.get_session_key(session_key_id)
        
        assert retrieved == session_key
    
    def test_get_nonexistent_session_key(self):
        """Test retrieving a non-existent session key."""
        crypto = TEECrypto()
        
        result = crypto.get_session_key("nonexistent-key-id")
        
        assert result is None
    
    def test_derive_key_from_shared_secret(self):
        """Test key derivation from shared secret."""
        shared_secret = secrets.token_bytes(32)
        
        key1 = TEECrypto.derive_key_from_shared_secret(shared_secret)
        key2 = TEECrypto.derive_key_from_shared_secret(shared_secret)
        
        assert len(key1) == KEY_SIZE
        assert key1 == key2  # Deterministic
    
    def test_derive_key_with_different_info(self):
        """Test that different info produces different keys."""
        shared_secret = secrets.token_bytes(32)
        
        key1 = TEECrypto.derive_key_from_shared_secret(shared_secret, info=b"info1")
        key2 = TEECrypto.derive_key_from_shared_secret(shared_secret, info=b"info2")
        
        assert key1 != key2


class TestEncryptDecrypt:
    """Tests for encryption and decryption."""
    
    @pytest.fixture
    def sample_payload(self):
        """Create a sample task payload."""
        return TaskPayload(
            reference_video_url="https://example.com/video.mp4",
            storage_credentials=StorageCredentials(
                endpoint_url="https://minio.example.com",
                access_key="AKIAIOSFODNN7EXAMPLE",
                secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                bucket="test-bucket",
                region="us-east-1",
            ),
            result_object_key="results/test-output.mp4",
            task_type="upscaling",
            task_params={"task_type": "HD24K"},
        )
    
    @pytest.fixture
    def crypto_with_key(self):
        """Create crypto instance with a session key."""
        crypto = TEECrypto()
        session_key_id, session_key = crypto.generate_session_key()
        return crypto, session_key_id, session_key
    
    def test_encrypt_payload(self, sample_payload, crypto_with_key):
        """Test payload encryption."""
        crypto, session_key_id, session_key = crypto_with_key
        
        encrypted = TEECrypto.encrypt_payload(
            sample_payload, session_key, session_key_id
        )
        
        assert isinstance(encrypted, EncryptedPayload)
        assert len(encrypted.encrypted_blob) > 0
        assert len(encrypted.nonce) == NONCE_SIZE
        assert encrypted.session_key_id == session_key_id
    
    def test_decrypt_payload(self, sample_payload, crypto_with_key):
        """Test payload decryption."""
        crypto, session_key_id, session_key = crypto_with_key
        
        # Encrypt
        encrypted = TEECrypto.encrypt_payload(
            sample_payload, session_key, session_key_id
        )
        
        # Decrypt
        decrypted = crypto.decrypt_payload(encrypted, session_key)
        
        assert decrypted.reference_video_url == sample_payload.reference_video_url
        assert decrypted.storage_credentials.access_key == sample_payload.storage_credentials.access_key
        assert decrypted.result_object_key == sample_payload.result_object_key
        assert decrypted.task_type == sample_payload.task_type
        assert decrypted.task_params == sample_payload.task_params
    
    def test_decrypt_with_wrong_key(self, sample_payload, crypto_with_key):
        """Test decryption with wrong key fails."""
        crypto, session_key_id, session_key = crypto_with_key
        
        # Encrypt
        encrypted = TEECrypto.encrypt_payload(
            sample_payload, session_key, session_key_id
        )
        
        # Try to decrypt with wrong key
        wrong_key = secrets.token_bytes(KEY_SIZE)
        
        with pytest.raises(ValueError, match="Failed to decrypt"):
            crypto.decrypt_payload(encrypted, wrong_key)
    
    def test_decrypt_with_stored_key(self, sample_payload, crypto_with_key):
        """Test decryption using stored session key lookup."""
        crypto, session_key_id, session_key = crypto_with_key
        
        # Encrypt
        encrypted = TEECrypto.encrypt_payload(
            sample_payload, session_key, session_key_id
        )
        
        # Decrypt without providing key (should look up by ID)
        decrypted = crypto.decrypt_payload(encrypted)
        
        assert decrypted.reference_video_url == sample_payload.reference_video_url
    
    def test_encrypted_payload_serialization(self, sample_payload, crypto_with_key):
        """Test EncryptedPayload to/from dict."""
        crypto, session_key_id, session_key = crypto_with_key
        
        encrypted = TEECrypto.encrypt_payload(
            sample_payload, session_key, session_key_id
        )
        
        # Serialize
        data = encrypted.to_dict()
        
        assert isinstance(data["encrypted_blob"], str)  # Base64
        assert isinstance(data["nonce"], str)  # Base64
        assert data["session_key_id"] == session_key_id
        
        # Deserialize
        restored = EncryptedPayload.from_dict(data)
        
        assert restored.encrypted_blob == encrypted.encrypted_blob
        assert restored.nonce == encrypted.nonce
        assert restored.session_key_id == encrypted.session_key_id


class TestChecksums:
    """Tests for checksum computation."""
    
    def test_compute_checksum(self):
        """Test data checksum computation."""
        data = b"test data for checksum"
        
        checksum = TEECrypto.compute_checksum(data)
        
        assert len(checksum) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in checksum)
    
    def test_checksum_deterministic(self):
        """Test that checksum is deterministic."""
        data = b"same data"
        
        checksum1 = TEECrypto.compute_checksum(data)
        checksum2 = TEECrypto.compute_checksum(data)
        
        assert checksum1 == checksum2
    
    def test_different_data_different_checksum(self):
        """Test that different data produces different checksum."""
        data1 = b"data one"
        data2 = b"data two"
        
        checksum1 = TEECrypto.compute_checksum(data1)
        checksum2 = TEECrypto.compute_checksum(data2)
        
        assert checksum1 != checksum2


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_encrypt_task_for_miner(self):
        """Test the convenience function for encrypting tasks."""
        session_key = secrets.token_bytes(KEY_SIZE)
        session_key_id = secrets.token_hex(16)
        
        storage_creds = StorageCredentials(
            endpoint_url="https://minio.example.com",
            access_key="testkey",
            secret_key="testsecret",
            bucket="test-bucket",
        )
        
        encrypted = encrypt_task_for_miner(
            video_url="https://example.com/video.mp4",
            storage_credentials=storage_creds,
            result_object_key="results/output.mp4",
            task_type="compression",
            task_params={"vmaf_threshold": 90.0},
            session_key=session_key,
            session_key_id=session_key_id,
        )
        
        assert isinstance(encrypted, EncryptedPayload)
        
        # Verify we can decrypt
        crypto = TEECrypto()
        decrypted = crypto.decrypt_payload(encrypted, session_key)
        
        assert decrypted.reference_video_url == "https://example.com/video.mp4"
        assert decrypted.task_type == "compression"
        assert decrypted.task_params["vmaf_threshold"] == 90.0


class TestStorageCredentials:
    """Tests for StorageCredentials dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        creds = StorageCredentials(
            endpoint_url="https://minio.example.com",
            access_key="testkey",
            secret_key="testsecret",
            bucket="test-bucket",
            region="us-west-2",
        )
        
        data = creds.to_dict()
        
        assert data["endpoint_url"] == "https://minio.example.com"
        assert data["bucket"] == "test-bucket"
        assert data["region"] == "us-west-2"
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "endpoint_url": "https://s3.amazonaws.com",
            "access_key": "key",
            "secret_key": "secret",
            "bucket": "mybucket",
            "region": "eu-west-1",
        }
        
        creds = StorageCredentials.from_dict(data)
        
        assert creds.endpoint_url == "https://s3.amazonaws.com"
        assert creds.region == "eu-west-1"


class TestTaskPayload:
    """Tests for TaskPayload dataclass."""
    
    def test_round_trip(self):
        """Test serialization round-trip."""
        payload = TaskPayload(
            reference_video_url="https://example.com/video.mp4",
            storage_credentials=StorageCredentials(
                endpoint_url="https://minio.example.com",
                access_key="key",
                secret_key="secret",
                bucket="bucket",
            ),
            result_object_key="results/output.mp4",
            task_type="upscaling",
            task_params={"scale": 2},
        )
        
        data = payload.to_dict()
        restored = TaskPayload.from_dict(data)
        
        assert restored.reference_video_url == payload.reference_video_url
        assert restored.storage_credentials.bucket == payload.storage_credentials.bucket
        assert restored.task_params["scale"] == 2
