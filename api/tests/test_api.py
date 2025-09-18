"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(authenticated_client, mock_llm_dependencies):
    """Test the health endpoint (no auth required)."""
    mock_llm, mock_agent = mock_llm_dependencies
    resp = authenticated_client.get('/api/v1/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'status' in data


def test_chat_complete_authenticated(
    authenticated_client,
    auth_headers,
    mock_user_dependencies,
    mock_llm_dependencies
):
    """Test authenticated chat completion endpoint."""
    mock_llm, mock_agent = mock_llm_dependencies

    resp = authenticated_client.post(
        '/api/v1/chat/complete',
        json={'message': 'hello', 'project_id': 'test'},
        headers=auth_headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 'response' in data


def test_chat_streaming_authenticated(
    authenticated_client,
    auth_headers,
    mock_user_dependencies,
    mock_llm_dependencies
):
    """Test authenticated streaming chat endpoint."""
    mock_llm, mock_agent = mock_llm_dependencies

    resp = authenticated_client.post(
        '/api/v1/chat',
        json={'message': 'hello', 'project_id': 'test'},
        headers=auth_headers
    )
    assert resp.status_code == 200
    # For streaming endpoints, we just check it doesn't error


def test_chat_unauthenticated(authenticated_client, mock_llm_dependencies):
    """Test that chat endpoints require authentication."""
    mock_llm, mock_agent = mock_llm_dependencies
    resp = authenticated_client.post(
        '/api/v1/chat/complete',
        json={'message': 'hello', 'project_id': 'test'}
    )
    assert resp.status_code == 401


def test_tools_endpoint_authenticated(
    authenticated_client,
    auth_headers,
    mock_user_dependencies
):
    """Test tools endpoint with authentication."""
    resp = authenticated_client.get('/api/v1/tools', headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert 'tools' in data