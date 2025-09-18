"""Authentication and network security middleware."""

import logging
from typing import Optional, Callable
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request

from .models import UserResponse, TokenData
from .security import verify_token
from .network import get_client_ip, is_ip_allowed, get_network_info
from .database import auth_db

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


class NetworkSecurityMiddleware:
    """Middleware for network-based access control."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request information
        request = Request(scope, receive)
        client_ip = get_client_ip(request)
        path = scope.get("path", "")

        # Skip network validation for health checks and static files
        if path in ["/ping", "/docs", "/redoc", "/openapi.json"] or path.startswith("/static/"):
            await self.app(scope, receive, send)
            return

        # Validate network access
        if not is_ip_allowed(client_ip):
            network_info = get_network_info(client_ip)
            logger.warning(f"Access denied from IP {client_ip}: {network_info}")

            response_body = b'{"detail":"Access denied: Invalid network location"}'
            response = {
                "type": "http.response.start",
                "status": 403,
                "headers": [[b"content-type", b"application/json"]],
            }
            await send(response)
            await send({"type": "http.response.body", "body": response_body})
            return

        # Log allowed access
        logger.debug(f"Network access allowed from IP {client_ip}")
        await self.app(scope, receive, send)


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserResponse]:
    """Get current user from token (optional - doesn't raise exception if no token).

    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials

    Returns:
        User data if authenticated, None otherwise
    """
    if not credentials:
        return None

    try:
        # Verify token
        token_data = verify_token(credentials.credentials)
        if not token_data or not token_data.user_id:
            return None

        # Get user from database
        user = await auth_db.get_user_by_id(token_data.user_id)
        if not user or not user.active:
            return None

        # Verify session exists and is valid
        session = await auth_db.get_session(credentials.credentials)
        if not session:
            return None

        return user

    except Exception as e:
        logger.warning(f"Token validation error: {e}")
        return None


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserResponse:
    """Get current authenticated user (required).

    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials

    Returns:
        User data

    Raises:
        HTTPException: If not authenticated
    """
    user = await get_current_user_optional(request, credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_admin_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Get current user and verify admin role.

    Args:
        current_user: Current authenticated user

    Returns:
        User data

    Raises:
        HTTPException: If not admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_network_access(request: Request) -> str:
    """Dependency to validate network access."""
    client_ip = get_client_ip(request)
    if not is_ip_allowed(client_ip):
        network_info = get_network_info(client_ip)
        logger.warning(f"Access denied from IP {client_ip}: {network_info}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Invalid network location"
        )
    return client_ip


def create_auth_dependency(require_auth: bool = True, require_admin: bool = False):
    """Create authentication dependency with flexible requirements.

    Args:
        require_auth: Whether authentication is required
        require_admin: Whether admin role is required

    Returns:
        FastAPI dependency function
    """
    if require_admin:
        return Depends(get_current_admin_user)
    elif require_auth:
        return Depends(get_current_user)
    else:
        return Depends(get_current_user_optional)


class AuthenticationState:
    """Helper class to track authentication state."""

    def __init__(self):
        self.initialized = False
        self.admin_exists = False

    async def check_admin_exists(self) -> bool:
        """Check if any admin user exists."""
        try:
            users = await auth_db.list_users()
            self.admin_exists = any(user.role == "admin" for user in users)
            return self.admin_exists
        except Exception:
            return False

    async def initialize(self) -> None:
        """Initialize authentication state."""
        try:
            await auth_db.initialize()
            await self.check_admin_exists()
            self.initialized = True
            logger.info("Authentication system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize authentication: {e}")
            raise


# Global authentication state
auth_state = AuthenticationState()