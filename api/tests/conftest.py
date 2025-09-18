"""Test configuration and fixtures for API tests."""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from api.main import app
from auth.models import UserResponse, UserRole
from auth.security import create_access_token


@pytest.fixture
def mock_network_bypass():
    """Mock network access control to allow TestClient access."""
    with patch('auth.middleware.is_ip_allowed', return_value=True):
        with patch('auth.network.get_client_ip', return_value='127.0.0.1'):
            yield


@pytest.fixture
def test_user():
    """Create a test user for authentication."""
    return UserResponse(
        id=1,
        username="testuser",
        email="test@example.com",
        role=UserRole.USER,
        active=True,
        created_at=datetime.now()
    )


@pytest.fixture
def test_admin():
    """Create a test admin user for authentication."""
    return UserResponse(
        id=2,
        username="testadmin",
        email="admin@example.com",
        role=UserRole.ADMIN,
        active=True,
        created_at=datetime.now()
    )


@pytest.fixture
def user_token(test_user):
    """Create a JWT token for the test user."""
    return create_access_token(
        data={
            "sub": test_user.username,
            "user_id": test_user.id,
            "role": test_user.role.value
        },
        expires_delta=timedelta(hours=1)
    )


@pytest.fixture
def admin_token(test_admin):
    """Create a JWT token for the test admin."""
    return create_access_token(
        data={
            "sub": test_admin.username,
            "user_id": test_admin.id,
            "role": test_admin.role.value
        },
        expires_delta=timedelta(hours=1)
    )


@pytest.fixture
def auth_headers(user_token):
    """Create authentication headers for requests."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def admin_headers(admin_token):
    """Create admin authentication headers for requests."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def authenticated_client(mock_network_bypass):
    """Create a TestClient with network bypass."""
    return TestClient(app)


@pytest.fixture
def mock_auth_db():
    """Mock the auth database for testing."""
    with patch('auth.database.auth_db') as mock_db:
        # Mock user retrieval
        mock_db.get_user_by_username.return_value = None
        mock_db.get_user_by_id.return_value = None
        mock_db.create_session.return_value = None
        mock_db.delete_session.return_value = None
        yield mock_db


@pytest.fixture
def mock_user_dependencies(test_user, mock_auth_db):
    """Mock user authentication dependencies."""
    with patch('auth.middleware.get_current_user', return_value=test_user):
        with patch('auth.middleware.get_current_user_optional', return_value=test_user):
            yield


@pytest.fixture
def mock_admin_dependencies(test_admin, mock_auth_db):
    """Mock admin authentication dependencies."""
    with patch('auth.middleware.get_current_user', return_value=test_admin):
        with patch('auth.middleware.get_current_admin_user', return_value=test_admin):
            yield


@pytest.fixture
def mock_llm_dependencies():
    """Mock LLM and agent dependencies for API testing."""
    with patch('api.router._llm_engine') as mock_llm_var:
        with patch('api.router._agent_instance') as mock_agent_var:
            # Create mock objects
            mock_llm = Mock()
            mock_agent = Mock()

            # Configure mock LLM
            mock_llm.generate.return_value = iter(["Hello", " from", " test", " LLM"])

            # Configure mock agent
            async def mock_chat(message):
                for chunk in ["Test", " response", " from", " agent"]:
                    yield chunk

            mock_agent.chat.return_value = mock_chat(None)

            # Set the module-level variables to our mocks
            mock_llm_var.return_value = mock_llm
            mock_agent_var.return_value = mock_agent

            # Also patch the dependency functions directly for reliability
            with patch('api.router.get_agent', return_value=mock_agent):
                with patch('api.router.get_llm_engine', return_value=mock_llm):
                    yield mock_llm, mock_agent