"""
Tests for SGX Attestation Verification

Tests the attestation verifier, policy enforcement, and quote parsing.
"""

import pytest
import json

from vidaio_subnet_core.tee.attestation_verifier import (
    AttestationVerifier,
    AttestationPolicy,
    AttestationResult,
    EnclaveReport,
    create_mock_quote,
    DEFAULT_MINER_POLICY,
)


class TestEnclaveReport:
    """Tests for EnclaveReport dataclass."""
    
    def test_mrenclave_hex(self):
        """Test MRENCLAVE hex conversion."""
        mrenclave = bytes.fromhex("a" * 64)
        report = EnclaveReport(
            mrenclave=mrenclave,
            mrsigner=bytes(32),
            isv_prod_id=1,
            isv_svn=1,
            attributes=bytes(16),
            is_debug=False,
        )
        
        assert report.mrenclave_hex() == "a" * 64
    
    def test_to_dict(self):
        """Test serialization to dict."""
        report = EnclaveReport(
            mrenclave=bytes.fromhex("ab" * 32),
            mrsigner=bytes.fromhex("cd" * 32),
            isv_prod_id=2,
            isv_svn=3,
            attributes=bytes.fromhex("ef" * 16),
            is_debug=True,
        )
        
        data = report.to_dict()
        
        assert data["mrenclave"] == "ab" * 32
        assert data["mrsigner"] == "cd" * 32
        assert data["isv_prod_id"] == 2
        assert data["isv_svn"] == 3
        assert data["is_debug"] is True
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "mrenclave": "12" * 32,
            "mrsigner": "34" * 32,
            "isv_prod_id": 5,
            "isv_svn": 6,
            "attributes": "00" * 16,
            "is_debug": False,
        }
        
        report = EnclaveReport.from_dict(data)
        
        assert report.mrenclave_hex() == "12" * 32
        assert report.isv_prod_id == 5
        assert report.is_debug is False


class TestAttestationPolicy:
    """Tests for AttestationPolicy."""
    
    def test_basic_policy(self):
        """Test basic policy creation."""
        policy = AttestationPolicy(
            allowed_mrenclaves=["a" * 64, "b" * 64],
            allow_debug=False,
            min_isv_svn=2,
        )
        
        assert len(policy.allowed_mrenclaves) == 2
        assert policy.allow_debug is False
        assert policy.min_isv_svn == 2
    
    def test_to_dict(self):
        """Test policy serialization."""
        policy = AttestationPolicy(
            allowed_mrenclaves=["x" * 64],
            allowed_mrsigners=["y" * 64],
            min_isv_svn=5,
            allow_debug=True,
        )
        
        data = policy.to_dict()
        
        assert data["allowed_mrenclaves"] == ["x" * 64]
        assert data["allowed_mrsigners"] == ["y" * 64]
        assert data["min_isv_svn"] == 5
        assert data["allow_debug"] is True
    
    def test_from_dict(self):
        """Test policy deserialization."""
        data = {
            "allowed_mrenclaves": ["z" * 64],
            "min_isv_svn": 3,
        }
        
        policy = AttestationPolicy.from_dict(data)
        
        assert policy.allowed_mrenclaves == ["z" * 64]
        assert policy.min_isv_svn == 3
        assert policy.allow_debug is False  # Default


class TestMockQuote:
    """Tests for mock quote creation."""
    
    def test_create_mock_quote(self):
        """Test creating a mock quote."""
        mrenclave = "ab" * 32
        
        quote = create_mock_quote(
            mrenclave=mrenclave,
            isv_prod_id=1,
            isv_svn=2,
            is_debug=False,
        )
        
        assert isinstance(quote, bytes)
        
        # Parse and verify
        data = json.loads(quote.decode('utf-8'))
        assert data["mrenclave"] == mrenclave
        assert data["isv_prod_id"] == 1
        assert data["isv_svn"] == 2
        assert data["is_debug"] is False


class TestAttestationVerifier:
    """Tests for AttestationVerifier."""
    
    @pytest.fixture
    def simple_policy(self):
        """Create a simple test policy."""
        return AttestationPolicy(
            allowed_mrenclaves=["a" * 64, "b" * 64],
            allow_debug=True,
            min_isv_svn=1,
        )
    
    @pytest.fixture
    def verifier(self, simple_policy):
        """Create a verifier with simple policy."""
        return AttestationVerifier(simple_policy)
    
    def test_verify_valid_quote(self, verifier):
        """Test verifying a valid quote."""
        quote = create_mock_quote(
            mrenclave="a" * 64,
            isv_svn=1,
            is_debug=True,
        )
        
        result, report = verifier.verify_quote(quote)
        
        assert result == AttestationResult.SUCCESS
        assert report is not None
        assert report.mrenclave_hex() == "a" * 64
    
    def test_verify_mrenclave_mismatch(self, verifier):
        """Test rejecting quote with wrong MRENCLAVE."""
        quote = create_mock_quote(
            mrenclave="c" * 64,  # Not in allowed list
            isv_svn=1,
        )
        
        result, report = verifier.verify_quote(quote)
        
        assert result == AttestationResult.MRENCLAVE_MISMATCH
        assert report is not None
    
    def test_reject_debug_enclave(self):
        """Test rejecting debug enclave when not allowed."""
        policy = AttestationPolicy(
            allowed_mrenclaves=["a" * 64],
            allow_debug=False,  # No debug allowed
        )
        verifier = AttestationVerifier(policy)
        
        quote = create_mock_quote(
            mrenclave="a" * 64,
            is_debug=True,  # Debug enclave
        )
        
        result, report = verifier.verify_quote(quote)
        
        assert result == AttestationResult.DEBUG_ENCLAVE
    
    def test_reject_low_svn(self):
        """Test rejecting enclave with low SVN."""
        policy = AttestationPolicy(
            allowed_mrenclaves=["a" * 64],
            min_isv_svn=5,  # Require SVN >= 5
            allow_debug=True,
        )
        verifier = AttestationVerifier(policy)
        
        quote = create_mock_quote(
            mrenclave="a" * 64,
            isv_svn=3,  # Too low
        )
        
        result, report = verifier.verify_quote(quote)
        
        assert result == AttestationResult.VERIFICATION_FAILED
    
    def test_verify_with_mrsigner_check(self):
        """Test MRSIGNER verification."""
        policy = AttestationPolicy(
            allowed_mrenclaves=["a" * 64],
            allowed_mrsigners=["b" * 64],  # Specific signer required
            allow_debug=True,
        )
        verifier = AttestationVerifier(policy)
        
        # Wrong signer
        quote = create_mock_quote(
            mrenclave="a" * 64,
            mrsigner="c" * 64,  # Wrong signer
        )
        
        result, report = verifier.verify_quote(quote)
        
        assert result == AttestationResult.MRSIGNER_MISMATCH
        
        # Correct signer
        quote = create_mock_quote(
            mrenclave="a" * 64,
            mrsigner="b" * 64,  # Correct signer
        )
        
        result, report = verifier.verify_quote(quote)
        
        assert result == AttestationResult.SUCCESS
    
    def test_extract_enclave_report(self, verifier):
        """Test extracting enclave report without full verification."""
        quote = create_mock_quote(
            mrenclave="d" * 64,
            isv_prod_id=7,
        )
        
        report = verifier.extract_enclave_report(quote)
        
        assert report is not None
        assert report.mrenclave_hex() == "d" * 64
        assert report.isv_prod_id == 7
    
    def test_invalid_quote(self, verifier):
        """Test handling invalid quote."""
        invalid_quote = b"not a valid quote"
        
        result, report = verifier.verify_quote(invalid_quote)
        
        assert result == AttestationResult.INVALID_QUOTE
        assert report is None


class TestDefaultMinerPolicy:
    """Tests for the default miner policy."""
    
    def test_default_policy_exists(self):
        """Test that default policy is defined."""
        assert DEFAULT_MINER_POLICY is not None
        assert len(DEFAULT_MINER_POLICY.allowed_mrenclaves) > 0
    
    def test_default_policy_allows_debug(self):
        """Test that default policy allows debug for development."""
        # Note: This should be False in production
        assert DEFAULT_MINER_POLICY.allow_debug is True
