"""Authentication API endpoints."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.security import HTTPAuthorizationCredentials

from .models import (
    UserCreate, UserResponse, UserLogin, Token, UserRole
)
from .security import (
    verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .network import get_client_ip, get_network_info
from .database import auth_db
from .middleware import (
    get_current_user, get_current_admin_user, get_current_user_optional,
    require_network_access, auth_state, security
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    client_ip: str = Depends(require_network_access)
) -> Token:
    """Authenticate user and return access token.

    Args:
        request: FastAPI request object
        username: Username
        password: Password
        client_ip: Client IP address (from dependency)

    Returns:
        JWT access token

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Get user from database
        user = await auth_db.get_user_by_username(username)
        if not user or not user.active:
            logger.warning(f"Login attempt for non-existent/inactive user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )

        # Verify password
        if not verify_password(password, user.password_hash):
            logger.warning(f"Failed login attempt for user: {username} from {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )

        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "role": user.role.value
            },
            expires_delta=access_token_expires
        )

        # Create session
        expires_at = datetime.utcnow() + access_token_expires
        user_agent = request.headers.get("User-Agent")
        await auth_db.create_session(
            user_id=user.id,
            token=access_token,
            expires_at=expires_at,
            ip_address=client_ip,
            user_agent=user_agent
        )

        logger.info(f"Successful login for user: {username} from {client_ip}")

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, str]:
    """Logout user by invalidating token.

    Args:
        credentials: HTTP Bearer credentials
        current_user: Current authenticated user

    Returns:
        Success message
    """
    try:
        if credentials:
            # Delete session
            await auth_db.delete_session(credentials.credentials)
            logger.info(f"User {current_user.username} logged out")

        return {"message": "Successfully logged out"}

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service error"
        )


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    current_admin: UserResponse = Depends(get_current_admin_user),
    client_ip: str = Depends(require_network_access)
) -> UserResponse:
    """Register a new user (admin only).

    Args:
        user_data: User creation data
        current_admin: Current admin user
        client_ip: Client IP address

    Returns:
        Created user data

    Raises:
        HTTPException: If registration fails
    """
    try:
        # Create user
        user = await auth_db.create_user(user_data)
        logger.info(f"User {user.username} created by admin {current_admin.username} from {client_ip}")
        return user

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration service error"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User information
    """
    return current_user


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_admin: UserResponse = Depends(get_current_admin_user)
) -> List[UserResponse]:
    """List all users (admin only).

    Args:
        current_admin: Current admin user

    Returns:
        List of users
    """
    try:
        users = await auth_db.list_users()
        return users

    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User service error"
        )


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    current_admin: UserResponse = Depends(get_current_admin_user)
) -> Dict[str, str]:
    """Deactivate a user account (admin only).

    Args:
        user_id: User ID to deactivate
        current_admin: Current admin user

    Returns:
        Success message

    Raises:
        HTTPException: If deactivation fails
    """
    try:
        # Prevent admin from deactivating themselves
        if user_id == current_admin.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )

        # Deactivate user
        success = await auth_db.update_user_active_status(user_id, False)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Delete all user sessions
        await auth_db.delete_user_sessions(user_id)

        logger.info(f"User {user_id} deactivated by admin {current_admin.username}")
        return {"message": "User deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User service error"
        )


@router.get("/status")
async def auth_status(
    request: Request,
    current_user: UserResponse = Depends(get_current_user_optional),
    client_ip: str = Depends(require_network_access)
) -> Dict[str, Any]:
    """Get authentication system status.

    Args:
        request: FastAPI request object
        current_user: Current user (if authenticated)
        client_ip: Client IP address

    Returns:
        Authentication status information
    """
    try:
        network_info = get_network_info(client_ip)

        status_info = {
            "authenticated": current_user is not None,
            "network_access": True,  # If we reach here, network access is allowed
            "network_info": network_info,
            "auth_required": auth_state.initialized,
            "admin_exists": auth_state.admin_exists,
        }

        if current_user:
            status_info["user"] = {
                "username": current_user.username,
                "role": current_user.role,
                "id": current_user.id
            }

        return status_info

    except Exception as e:
        logger.error(f"Auth status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Status service error"
        )


@router.post("/cleanup-sessions")
async def cleanup_expired_sessions(
    current_admin: UserResponse = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """Clean up expired sessions (admin only).

    Args:
        current_admin: Current admin user

    Returns:
        Cleanup results
    """
    try:
        deleted_count = await auth_db.delete_expired_sessions()
        logger.info(f"Cleaned up {deleted_count} expired sessions by admin {current_admin.username}")

        return {
            "message": "Expired sessions cleaned up",
            "deleted_sessions": deleted_count
        }

    except Exception as e:
        logger.error(f"Session cleanup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session cleanup error"
        )