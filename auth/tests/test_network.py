"""Tests for network security utilities."""

import pytest
import os
import ipaddress
from unittest.mock import Mock, patch
from fastapi import Request

from auth.network import (
    get_allowed_networks, get_client_ip, is_ip_allowed, is_localhost,
    is_private_network, get_network_info, validate_network_config
)


class TestNetworkConfiguration:
    """Test network configuration and setup."""

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_default_allowed_networks(self):
        """Test default network configuration."""
        # Need to reload to pick up env changes
        import importlib
        import auth.network
        importlib.reload(auth.network)

        networks = auth.network.get_allowed_networks()

        # Should include default private networks
        network_strings = [str(net) for net in networks]
        assert "127.0.0.0/8" in network_strings      # Localhost
        assert "10.0.0.0/8" in network_strings       # Private Class A
        assert "172.16.0.0/12" in network_strings    # Private Class B
        assert "192.168.0.0/16" in network_strings   # Private Class C

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": "203.0.113.0/24,198.51.100.0/24"
    })
    def test_custom_allowed_networks(self):
        """Test custom network configuration."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        networks = auth.network.get_allowed_networks()
        network_strings = [str(net) for net in networks]

        # Should include both default and custom networks
        assert "127.0.0.0/8" in network_strings      # Default
        assert "203.0.113.0/24" in network_strings   # Custom
        assert "198.51.100.0/24" in network_strings  # Custom

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "false",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_disabled_network_security(self):
        """Test with network security disabled."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        networks = auth.network.get_allowed_networks()
        # Should be empty when security is disabled and no custom networks
        assert len(networks) == 0

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "false",
        "AUTH_ALLOWED_NETWORKS": "invalid-network,192.168.1.0/24"
    })
    def test_invalid_network_handling(self):
        """Test handling of invalid network configurations."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        networks = auth.network.get_allowed_networks()
        network_strings = [str(net) for net in networks]

        # Should include only valid networks, skip invalid ones
        assert "192.168.1.0/24" in network_strings
        assert len([n for n in network_strings if "invalid" in n]) == 0


class TestClientIPExtraction:
    """Test client IP extraction from requests."""

    def test_direct_client_ip(self):
        """Test extracting IP from direct connection."""
        # Mock request with direct client
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"

        ip = get_client_ip(mock_request)
        assert ip == "192.168.1.100"

    def test_forwarded_for_header(self):
        """Test extracting IP from X-Forwarded-For header."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "X-Forwarded-For": "203.0.113.1, 192.168.1.1, 10.0.0.1"
        }
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # Should return first IP in chain
        ip = get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_real_ip_header(self):
        """Test extracting IP from X-Real-IP header."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "X-Real-IP": "203.0.113.42"
        }
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        ip = get_client_ip(mock_request)
        assert ip == "203.0.113.42"

    def test_header_priority(self):
        """Test header priority when multiple are present."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "X-Forwarded-For": "203.0.113.1",
            "X-Real-IP": "203.0.113.2"
        }
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # X-Forwarded-For should take priority
        ip = get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_no_client_info(self):
        """Test handling when no client info is available."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.client = None

        ip = get_client_ip(mock_request)
        assert ip == "unknown"

    def test_whitespace_trimming(self):
        """Test that IP addresses are properly trimmed."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "X-Forwarded-For": "  203.0.113.1  , 192.168.1.1  "
        }

        ip = get_client_ip(mock_request)
        assert ip == "203.0.113.1"  # Should be trimmed


class TestIPValidation:
    """Test IP address validation functions."""

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_localhost_validation(self):
        """Test localhost IP validation."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        assert auth.network.is_ip_allowed("127.0.0.1")
        assert auth.network.is_ip_allowed("127.0.0.100")
        assert auth.network.is_localhost("127.0.0.1")
        assert auth.network.is_localhost("127.1.2.3")

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_private_network_validation(self):
        """Test private network IP validation."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        # Class A private
        assert auth.network.is_ip_allowed("10.0.0.1")
        assert auth.network.is_ip_allowed("10.255.255.255")

        # Class B private
        assert auth.network.is_ip_allowed("172.16.0.1")
        assert auth.network.is_ip_allowed("172.31.255.255")

        # Class C private
        assert auth.network.is_ip_allowed("192.168.0.1")
        assert auth.network.is_ip_allowed("192.168.255.255")

        # Verify they're detected as private
        assert auth.network.is_private_network("10.0.0.1")
        assert auth.network.is_private_network("172.16.0.1")
        assert auth.network.is_private_network("192.168.1.1")

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_public_ip_rejection(self):
        """Test that public IPs are rejected with default config."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        # Common public IPs should be rejected
        assert not auth.network.is_ip_allowed("8.8.8.8")         # Google DNS
        assert not auth.network.is_ip_allowed("1.1.1.1")         # Cloudflare DNS
        assert not auth.network.is_ip_allowed("74.125.224.72")   # Google public IP
        assert not auth.network.is_ip_allowed("151.101.1.140")   # Reddit public IP

        # Verify they're not private
        assert not auth.network.is_private_network("8.8.8.8")
        assert not auth.network.is_private_network("74.125.224.72")

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "false",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_disabled_security_allows_all(self):
        """Test that disabled security allows all IPs."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        # Should allow both private and public when disabled
        assert auth.network.is_ip_allowed("127.0.0.1")
        assert auth.network.is_ip_allowed("192.168.1.1")
        assert auth.network.is_ip_allowed("8.8.8.8")
        assert auth.network.is_ip_allowed("203.0.113.1")

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": "74.125.224.0/24"
    })
    def test_custom_network_validation(self):
        """Test validation against custom network ranges."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        # Custom network should be allowed
        assert auth.network.is_ip_allowed("74.125.224.1")
        assert auth.network.is_ip_allowed("74.125.224.254")

        # Outside custom network should be rejected (even if public)
        assert not auth.network.is_ip_allowed("74.125.225.1")
        assert not auth.network.is_ip_allowed("8.8.8.8")

        # Private networks should still be allowed
        assert auth.network.is_ip_allowed("192.168.1.1")

    def test_invalid_ip_handling(self):
        """Test handling of invalid IP addresses."""
        assert not is_ip_allowed("invalid-ip")
        assert not is_ip_allowed("999.999.999.999")
        assert not is_ip_allowed("192.168.1")  # Incomplete
        assert not is_ip_allowed("")
        assert not is_ip_allowed("192.168.1.1.1")  # Too many octets

        # Invalid IPs should not be localhost or private
        assert not is_localhost("invalid-ip")
        assert not is_private_network("invalid-ip")

    def test_edge_case_ips(self):
        """Test edge case IP addresses."""
        # Broadcast addresses
        assert is_private_network("192.168.1.255")  # Network broadcast
        assert is_localhost("127.255.255.255")      # Localhost range

        # Network addresses
        assert is_private_network("192.168.1.0")    # Network address
        assert is_localhost("127.0.0.0")            # Localhost network


class TestNetworkInfo:
    """Test network information and analysis functions."""

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": "203.0.113.0/24"
    })
    def test_network_info_localhost(self):
        """Test network info for localhost addresses."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        info = auth.network.get_network_info("127.0.0.1")

        assert info["ip"] == "127.0.0.1"
        assert info["allowed"] is True
        assert info["localhost"] is True
        assert info["private"] is True  # Localhost is considered private
        assert info["network_type"] == "loopback"
        assert "matched_network" in info

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_network_info_private(self):
        """Test network info for private addresses."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        info = auth.network.get_network_info("192.168.1.100")

        assert info["ip"] == "192.168.1.100"
        assert info["allowed"] is True
        assert info["localhost"] is False
        assert info["private"] is True
        assert info["network_type"] == "private"
        assert info["matched_network"] == "192.168.0.0/16"

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_network_info_public(self):
        """Test network info for public addresses."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        info = auth.network.get_network_info("8.8.8.8")

        assert info["ip"] == "8.8.8.8"
        assert info["allowed"] is False
        assert info["localhost"] is False
        assert info["private"] is False
        assert info["network_type"] == "public"
        assert "matched_network" not in info

    def test_network_info_invalid(self):
        """Test network info for invalid addresses."""
        info = get_network_info("invalid-ip")

        assert info["ip"] == "invalid-ip"
        assert info["allowed"] is False
        assert info["localhost"] is False
        assert info["private"] is False
        assert info["network_type"] == "invalid"

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": "74.125.224.0/24"
    })
    def test_network_info_custom_range(self):
        """Test network info for custom allowed ranges."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        info = auth.network.get_network_info("74.125.224.50")

        assert info["ip"] == "74.125.224.50"
        assert info["allowed"] is True
        assert info["localhost"] is False
        assert info["private"] is False
        assert info["network_type"] == "public"  # Public IP in custom range
        assert info["matched_network"] == "74.125.224.0/24"


class TestNetworkConfigValidation:
    """Test network configuration validation."""

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": "74.125.224.0/24,151.101.1.0/24"
    })
    def test_config_validation_enabled(self):
        """Test configuration validation with security enabled."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        config = auth.network.validate_network_config()

        assert config["require_local_network"] is True
        assert config["security_level"] == "high"
        assert len(config["allowed_networks"]) >= 6  # 4 default + 2 custom
        assert "74.125.224.0/24" in config["allowed_networks"]
        assert "151.101.1.0/24" in config["allowed_networks"]
        assert "127.0.0.0/8" in config["allowed_networks"]

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "false",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_config_validation_disabled(self):
        """Test configuration validation with security disabled."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        config = auth.network.validate_network_config()

        assert config["require_local_network"] is False
        assert config["security_level"] == "disabled"
        assert len(config["allowed_networks"]) == 0

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": "invalid-network,192.168.1.0/24,another-invalid"
    })
    def test_config_validation_mixed_valid_invalid(self):
        """Test configuration validation with mixed valid/invalid networks."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        config = auth.network.validate_network_config()

        # Should include default networks + valid custom network
        assert config["require_local_network"] is True
        assert config["security_level"] == "high"
        assert "192.168.1.0/24" in config["allowed_networks"]

        # Invalid networks should be in custom_networks but not allowed_networks
        assert "invalid-network" in config["custom_networks"]
        assert "invalid-network" not in config["allowed_networks"]


class TestNetworkSecurityIntegration:
    """Integration tests for network security components."""

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": "203.0.113.0/24"
    })
    def test_full_request_validation_cycle(self):
        """Test complete request validation cycle."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        # Mock request from allowed network
        mock_request = Mock(spec=Request)
        mock_request.headers = {"X-Forwarded-For": "192.168.1.100"}

        # Extract client IP
        client_ip = auth.network.get_client_ip(mock_request)
        assert client_ip == "192.168.1.100"

        # Validate IP
        is_allowed = auth.network.is_ip_allowed(client_ip)
        assert is_allowed is True

        # Get detailed info
        info = auth.network.get_network_info(client_ip)
        assert info["allowed"] is True
        assert info["network_type"] == "private"

    @patch.dict(os.environ, {
        "AUTH_REQUIRE_LOCAL_NETWORK": "true",
        "AUTH_ALLOWED_NETWORKS": ""
    })
    def test_blocked_request_cycle(self):
        """Test blocked request handling cycle."""
        import importlib
        import auth.network
        importlib.reload(auth.network)

        # Mock request from blocked network
        mock_request = Mock(spec=Request)
        mock_request.headers = {"X-Real-IP": "8.8.8.8"}

        client_ip = auth.network.get_client_ip(mock_request)
        assert client_ip == "8.8.8.8"

        is_allowed = auth.network.is_ip_allowed(client_ip)
        assert is_allowed is False

        info = auth.network.get_network_info(client_ip)
        assert info["allowed"] is False
        assert info["network_type"] == "public"

    def test_localhost_variants(self):
        """Test various localhost representations."""
        localhost_variants = [
            "127.0.0.1",
            "127.1.0.1",
            "127.255.255.254",
            "localhost",
            "::1"
        ]

        for variant in localhost_variants:
            if variant in ["localhost", "::1"]:
                # Special string handling
                assert is_localhost(variant)
            else:
                # IPv4 localhost
                assert is_localhost(variant)
                assert is_private_network(variant)