"""Authentication data models."""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles for authorization."""

    ADMIN = "admin"
    USER = "user"


class User(BaseModel):
    """Base user model."""

    username: str = Field(description="Unique username", min_length=3, max_length=50)
    email: Optional[EmailStr] = Field(default=None, description="User email address")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    active: bool = Field(default=True, description="Whether the user account is active")
    created_at: Optional[datetime] = Field(default=None, description="Account creation timestamp")


class UserInDB(User):
    """User model with database fields."""

    id: int = Field(description="User ID")
    password_hash: str = Field(description="Hashed password")


class UserCreate(BaseModel):
    """User creation request."""

    username: str = Field(description="Unique username", min_length=3, max_length=50)
    email: Optional[EmailStr] = Field(default=None, description="User email address")
    password: str = Field(description="User password", min_length=8, max_length=100)
    role: UserRole = Field(default=UserRole.USER, description="User role")


class UserResponse(BaseModel):
    """User response model (without sensitive data)."""

    id: int = Field(description="User ID")
    username: str = Field(description="Username")
    email: Optional[str] = Field(default=None, description="Email address")
    role: UserRole = Field(description="User role")
    active: bool = Field(description="Account status")
    created_at: datetime = Field(description="Account creation timestamp")


class UserLogin(BaseModel):
    """User login request."""

    username: str = Field(description="Username")
    password: str = Field(description="Password")


class Token(BaseModel):
    """JWT token response."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")


class TokenData(BaseModel):
    """JWT token payload data."""

    username: Optional[str] = Field(default=None, description="Username from token")
    user_id: Optional[int] = Field(default=None, description="User ID from token")
    role: Optional[UserRole] = Field(default=None, description="User role from token")


class Session(BaseModel):
    """User session model."""

    id: int = Field(description="Session ID")
    user_id: int = Field(description="User ID")
    token_hash: str = Field(description="Hashed token")
    expires_at: datetime = Field(description="Session expiration")
    created_at: datetime = Field(description="Session creation timestamp")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")