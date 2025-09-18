"""Global pytest configuration and shared fixtures."""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
from typing import Generator, Dict, Any


# Test environment configuration
TEST_SECRET_KEY = "test-secret-key-for-testing-only"
TEST_DB_PATH = "test_database.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment configuration."""
    test_env = {
        "AUTH_SECRET_KEY": TEST_SECRET_KEY,
        "AUTH_TOKEN_EXPIRE_MINUTES": "60",  # 1 hour for tests
        "AUTH_REQUIRE_LOCAL_NETWORK": "false",  # Disable for testing
        "AUTH_ALLOWED_NETWORKS": "",
        "PYTHONIOENCODING": "utf-8",
        "LLM_ENGINE_TYPE": "mock",  # Use mock engine for most tests
        "LLM_MODEL_PATH": "test-model",
    }

    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_database_path(temp_directory: Path) -> str:
    """Provide a temporary database path for testing."""
    return str(temp_directory / TEST_DB_PATH)


@pytest.fixture
def clean_test_environment():
    """Clean environment for tests that need isolation."""
    # Store original environment
    original_env = dict(os.environ)

    # Clean specific test variables
    test_vars = [
        "AUTH_SECRET_KEY",
        "AUTH_TOKEN_EXPIRE_MINUTES",
        "AUTH_REQUIRE_LOCAL_NETWORK",
        "AUTH_ALLOWED_NETWORKS",
        "LLM_ENGINE_TYPE",
        "LLM_MODEL_PATH"
    ]

    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls for shell command testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Mock command output",
            stderr=""
        )
        yield mock_run


@pytest.fixture
def mock_file_system():
    """Mock file system operations for path testing."""
    mock_functions = {}

    with patch('os.path.exists') as mock_exists, \
         patch('os.path.isfile') as mock_isfile, \
         patch('os.path.isdir') as mock_isdir, \
         patch('builtins.open', create=True) as mock_open:

        # Default behavior: files exist
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_isdir.return_value = False

        mock_functions.update({
            'exists': mock_exists,
            'isfile': mock_isfile,
            'isdir': mock_isdir,
            'open': mock_open
        })

        yield mock_functions


@pytest.fixture
def sample_test_data() -> Dict[str, Any]:
    """Provide sample data for testing."""
    return {
        "users": [
            {
                "id": 1,
                "username": "testuser",
                "email": "test@example.com",
                "role": "user",
                "active": True
            },
            {
                "id": 2,
                "username": "admin",
                "email": "admin@example.com",
                "role": "admin",
                "active": True
            }
        ],
        "conversations": [
            {
                "id": "conv_001",
                "user_id": 1,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ],
        "networks": {
            "localhost": "127.0.0.1",
            "private_class_a": "10.0.0.1",
            "private_class_b": "172.16.0.1",
            "private_class_c": "192.168.1.1",
            "public_google": "8.8.8.8",
            "public_cloudflare": "1.1.1.1"
        },
        "passwords": {
            "weak": "123456",
            "medium": "Password123",
            "strong": "SecureP@ssw0rd2024!"
        }
    }


@pytest.fixture
def performance_monitor():
    """Monitor test performance and resource usage."""
    import time
    import psutil

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    yield

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    duration = end_time - start_time
    memory_delta = end_memory - start_memory

    # Log performance metrics (could be stored for analysis)
    if duration > 5.0:  # Log slow tests
        print(f"\nSlow test detected: {duration:.2f}s")

    if memory_delta > 50 * 1024 * 1024:  # Log high memory usage (50MB)
        print(f"\nHigh memory usage: {memory_delta / 1024 / 1024:.2f}MB")


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "auth: marks tests as authentication-related"
    )
    config.addinivalue_line(
        "markers", "network: marks tests as network-related"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests as LLM engine-related"
    )
    config.addinivalue_line(
        "markers", "agent: marks tests as agent-related"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API-related"
    )


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify collected test items to add markers and organize tests."""
    for item in items:
        # Auto-mark tests based on file location
        test_file = str(item.fspath)

        if "auth/tests" in test_file:
            item.add_marker(pytest.mark.auth)
        elif "llm_server/tests" in test_file:
            item.add_marker(pytest.mark.llm)
        elif "agent/tests" in test_file:
            item.add_marker(pytest.mark.agent)
        elif "api/tests" in test_file:
            item.add_marker(pytest.mark.api)

        # Mark slow tests based on naming patterns
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.name or "full_cycle" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark network tests
        if "network" in item.name or "ip" in item.name:
            item.add_marker(pytest.mark.network)


# Test session hooks
def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("\nStarting 3dSt_Coder test suite...")

    # Ensure test directories exist
    test_dirs = [
        "llm_server/tests",
        "agent/tests",
        "api/tests",
        "auth/tests"
    ]

    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            print(f"Warning: Test directory {test_dir} not found")


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    if exitstatus == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\nTest session finished with exit code: {exitstatus}")


# Test report customization
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom terminal summary information."""
    terminalreporter.write_sep("=", "3dSt_Coder Test Summary")

    if hasattr(terminalreporter.stats, 'passed'):
        passed = len(terminalreporter.stats.get('passed', []))
        terminalreporter.write_line(f"Passed: {passed}")

    if hasattr(terminalreporter.stats, 'failed'):
        failed = len(terminalreporter.stats.get('failed', []))
        if failed > 0:
            terminalreporter.write_line(f"Failed: {failed}")

    if hasattr(terminalreporter.stats, 'skipped'):
        skipped = len(terminalreporter.stats.get('skipped', []))
        if skipped > 0:
            terminalreporter.write_line(f"Skipped: {skipped}")

    # Module-specific summary
    modules = ["auth", "llm_server", "agent", "api"]
    for module in modules:
        module_tests = [
            item for item in terminalreporter.stats.get('passed', [])
            if f"{module}/tests" in str(item.fspath)
        ]
        if module_tests:
            terminalreporter.write_line(f"{module}: {len(module_tests)} passed")