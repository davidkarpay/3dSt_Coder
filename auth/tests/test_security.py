"""Tests for authentication security utilities."""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import patch

from auth.security import (
    verify_password, get_password_hash, create_access_token,
    verify_token, hash_token, generate_secure_key, validate_password_strength
)
from auth.models import TokenData


class TestPasswordSecurity:
    """Test password hashing and verification."""

    def test_password_hashing_and_verification(self):
        """Test password hashing creates different hashes and verifies correctly."""
        password = "TestPassword123!"

        # Hash the password
        hashed1 = get_password_hash(password)
        hashed2 = get_password_hash(password)

        # Same password should produce different hashes (due to salt)
        assert hashed1 != hashed2
        assert hashed1 != password  # Should not store plaintext

        # Both hashes should verify correctly
        assert verify_password(password, hashed1)
        assert verify_password(password, hashed2)

        # Wrong password should not verify
        assert not verify_password("WrongPassword", hashed1)
        assert not verify_password("", hashed1)

    def test_password_strength_validation(self):
        """Test password strength validation function."""
        # Strong password
        strong_result = validate_password_strength("StrongP@ssw0rd!")
        assert strong_result["valid"]
        assert strong_result["strength"] == "strong"
        assert len(strong_result["errors"]) == 0
        assert strong_result["score"] == 5

        # Weak password - too short
        weak_result = validate_password_strength("123")
        assert not weak_result["valid"]
        assert "at least 8 characters" in str(weak_result["errors"])

        # Weak password - missing character types
        no_upper = validate_password_strength("password123!")
        assert "uppercase letter" in str(no_upper["errors"])

        no_lower = validate_password_strength("PASSWORD123!")
        assert "lowercase letter" in str(no_lower["errors"])

        no_digit = validate_password_strength("Password!")
        assert "number" in str(no_digit["errors"])

        no_special = validate_password_strength("Password123")
        assert "special character" in str(no_special["errors"])

        # Common password
        common_result = validate_password_strength("password")
        assert not common_result["valid"]
        assert "too common" in str(common_result["errors"])
        assert common_result["score"] == 0


class TestJWTTokens:
    """Test JWT token creation and verification."""

    @patch.dict(os.environ, {"AUTH_SECRET_KEY": "test-secret-key", "AUTH_TOKEN_EXPIRE_MINUTES": "60"})
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification cycle."""
        # Create token data
        user_data = {
            "sub": "testuser",
            "user_id": 123,
            "role": "user"
        }

        # Create token
        token = create_access_token(user_data)
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token
        token_data = verify_token(token)
        assert token_data is not None
        assert token_data.username == "testuser"
        assert token_data.user_id == 123
        assert token_data.role == "user"

    @patch.dict(os.environ, {"AUTH_SECRET_KEY": "test-secret-key"})
    def test_token_expiration(self):
        """Test that expired tokens are rejected."""
        user_data = {"sub": "testuser", "user_id": 123, "role": "user"}

        # Create token with very short expiration
        short_expiry = timedelta(seconds=-1)  # Already expired
        expired_token = create_access_token(user_data, short_expiry)

        # Should not verify
        token_data = verify_token(expired_token)
        assert token_data is None

    @patch.dict(os.environ, {"AUTH_SECRET_KEY": "test-secret-key"})
    def test_invalid_token_handling(self):
        """Test handling of invalid tokens."""
        # Completely invalid token
        assert verify_token("invalid.token.here") is None

        # Empty token
        assert verify_token("") is None

        # Token with wrong signature
        user_data = {"sub": "testuser", "user_id": 123, "role": "user"}
        token = create_access_token(user_data)

        # Modify the signature part
        parts = token.split('.')
        if len(parts) == 3:
            invalid_token = f"{parts[0]}.{parts[1]}.invalidsignature"
            assert verify_token(invalid_token) is None

    @patch.dict(os.environ, {"AUTH_SECRET_KEY": "test-secret-key"})
    def test_token_missing_required_fields(self):
        """Test tokens missing required fields."""
        # Token without 'sub' field
        incomplete_data = {"user_id": 123, "role": "user"}
        token = create_access_token(incomplete_data)

        # Should not verify due to missing username
        token_data = verify_token(token)
        assert token_data is None

    @patch.dict(os.environ, {"AUTH_SECRET_KEY": "test-secret-key"})
    def test_custom_expiration(self):
        """Test custom token expiration times."""
        user_data = {"sub": "testuser", "user_id": 123, "role": "user"}

        # Create token with 2 hour expiration
        custom_expiry = timedelta(hours=2)
        token = create_access_token(user_data, custom_expiry)

        # Should verify correctly
        token_data = verify_token(token)
        assert token_data is not None
        assert token_data.username == "testuser"


class TestTokenUtilities:
    """Test token utility functions."""

    def test_token_hashing(self):
        """Test token hashing for storage."""
        token = "sample.jwt.token"

        # Hash the token
        hash1 = hash_token(token)
        hash2 = hash_token(token)

        # Same token should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 character hex string

        # Different tokens should produce different hashes
        different_hash = hash_token("different.token.here")
        assert hash1 != different_hash

    def test_secure_key_generation(self):
        """Test secure key generation."""
        key1 = generate_secure_key()
        key2 = generate_secure_key()

        # Should generate different keys each time
        assert key1 != key2
        assert len(key1) > 0
        assert len(key2) > 0

        # Should be base64 encoded (no padding issues)
        import base64
        try:
            decoded1 = base64.urlsafe_b64decode(key1)
            decoded2 = base64.urlsafe_b64decode(key2)
            assert len(decoded1) == 32  # 256 bits
            assert len(decoded2) == 32  # 256 bits
        except Exception:
            pytest.fail("Generated keys should be valid base64")


class TestSecurityConfiguration:
    """Test security configuration and environment handling."""

    def test_default_configuration(self):
        """Test default security configuration values."""
        # Test with minimal environment
        with patch.dict(os.environ, {}, clear=True):
            from auth.security import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

            # Should have default values
            assert SECRET_KEY == "your-secret-key-change-in-production"
            assert ALGORITHM == "HS256"
            assert ACCESS_TOKEN_EXPIRE_MINUTES == 480  # 8 hours

    @patch.dict(os.environ, {
        "AUTH_SECRET_KEY": "custom-secret-key",
        "AUTH_TOKEN_EXPIRE_MINUTES": "120"
    })
    def test_custom_configuration(self):
        """Test custom security configuration from environment."""
        # Need to reload the module to pick up new env vars
        import importlib
        import auth.security
        importlib.reload(auth.security)

        assert auth.security.SECRET_KEY == "custom-secret-key"
        assert auth.security.ACCESS_TOKEN_EXPIRE_MINUTES == 120

    def test_token_data_model(self):
        """Test TokenData model creation and validation."""
        # Valid token data
        token_data = TokenData(
            username="testuser",
            user_id=123,
            role="admin"
        )

        assert token_data.username == "testuser"
        assert token_data.user_id == 123
        assert token_data.role == "admin"

        # Test with optional fields
        minimal_data = TokenData(username="user")
        assert minimal_data.username == "user"
        assert minimal_data.user_id is None
        assert minimal_data.role is None


class TestSecurityIntegration:
    """Integration tests for security components."""

    @patch.dict(os.environ, {"AUTH_SECRET_KEY": "integration-test-key"})
    def test_full_authentication_cycle(self):
        """Test complete authentication cycle."""
        # 1. Hash password for storage
        plain_password = "UserPassword123!"
        hashed_password = get_password_hash(plain_password)

        # 2. Verify password during login
        assert verify_password(plain_password, hashed_password)

        # 3. Create access token on successful login
        user_data = {
            "sub": "integrationuser",
            "user_id": 456,
            "role": "user"
        }
        access_token = create_access_token(user_data)

        # 4. Verify token for API requests
        token_data = verify_token(access_token)
        assert token_data.username == "integrationuser"
        assert token_data.user_id == 456
        assert token_data.role == "user"

        # 5. Hash token for session storage
        token_hash = hash_token(access_token)
        assert len(token_hash) == 64

    def test_password_security_edge_cases(self):
        """Test edge cases in password security."""
        # Empty password
        empty_hash = get_password_hash("")
        assert verify_password("", empty_hash)
        assert not verify_password("nonempty", empty_hash)

        # Very long password
        long_password = "a" * 1000
        long_hash = get_password_hash(long_password)
        assert verify_password(long_password, long_hash)

        # Unicode characters
        unicode_password = "密码Test123!@#"
        unicode_hash = get_password_hash(unicode_password)
        assert verify_password(unicode_password, unicode_hash)

    @patch.dict(os.environ, {"AUTH_SECRET_KEY": "edge-case-key"})
    def test_token_edge_cases(self):
        """Test edge cases in token handling."""
        # Token with unusual data types
        edge_data = {
            "sub": "edgeuser",
            "user_id": 0,  # Zero ID
            "role": "user",  # Valid role (empty string not allowed by enum)
            "custom_field": "custom_value"
        }

        token = create_access_token(edge_data)
        token_data = verify_token(token)

        assert token_data.username == "edgeuser"
        assert token_data.user_id == 0
        assert token_data.role.value == "user"