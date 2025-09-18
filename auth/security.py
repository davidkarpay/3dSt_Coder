"""Security utilities for authentication."""

import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from .models import TokenData


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("AUTH_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database

    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token.

    Args:
        token: JWT token to verify

    Returns:
        TokenData if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        role: str = payload.get("role")

        if username is None:
            return None

        token_data = TokenData(
            username=username,
            user_id=user_id,
            role=role
        )
        return token_data

    except JWTError:
        return None


def hash_token(token: str) -> str:
    """Create a hash of a token for storage.

    Args:
        token: JWT token to hash

    Returns:
        SHA256 hash of the token
    """
    return hashlib.sha256(token.encode()).hexdigest()


def generate_secure_key() -> str:
    """Generate a secure random key for JWT signing.

    Returns:
        Base64 encoded random key
    """
    import secrets
    import base64

    key = secrets.token_bytes(32)
    return base64.urlsafe_b64encode(key).decode()


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength requirements.

    Args:
        password: Password to validate

    Returns:
        Dictionary with validation results
    """
    errors = []
    score = 0

    # Length check
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    else:
        score += 1

    # Character type checks
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    if has_upper:
        score += 1
    else:
        errors.append("Password must contain at least one uppercase letter")

    if has_lower:
        score += 1
    else:
        errors.append("Password must contain at least one lowercase letter")

    if has_digit:
        score += 1
    else:
        errors.append("Password must contain at least one number")

    if has_special:
        score += 1
    else:
        errors.append("Password must contain at least one special character")

    # Common password check (basic)
    common_passwords = ["password", "123456", "admin", "user", "login"]
    if password.lower() in common_passwords:
        errors.append("Password is too common")
        score = 0

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "score": score,
        "strength": "weak" if score < 3 else "medium" if score < 5 else "strong"
    }