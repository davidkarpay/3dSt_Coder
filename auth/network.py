"""Network security utilities for IP validation."""

import os
import ipaddress
from typing import List, Optional, Union
from fastapi import Request


# Default allowed networks (private IP ranges)
DEFAULT_ALLOWED_NETWORKS = [
    "127.0.0.0/8",      # Localhost
    "10.0.0.0/8",       # Private Class A
    "172.16.0.0/12",    # Private Class B
    "192.168.0.0/16",   # Private Class C
]

# Custom networks from environment
CUSTOM_NETWORKS = os.getenv("AUTH_ALLOWED_NETWORKS", "").split(",")
CUSTOM_NETWORKS = [net.strip() for net in CUSTOM_NETWORKS if net.strip()]

# Whether to require local network access
REQUIRE_LOCAL_NETWORK = os.getenv("AUTH_REQUIRE_LOCAL_NETWORK", "true").lower() == "true"


def get_allowed_networks() -> List[ipaddress.IPv4Network]:
    """Get list of allowed network ranges.

    Returns:
        List of IPv4Network objects representing allowed networks
    """
    networks = []

    # Add default networks if local network requirement is enabled
    if REQUIRE_LOCAL_NETWORK:
        for network_str in DEFAULT_ALLOWED_NETWORKS:
            try:
                networks.append(ipaddress.IPv4Network(network_str))
            except ipaddress.AddressValueError:
                continue

    # Add custom networks from configuration
    for network_str in CUSTOM_NETWORKS:
        try:
            networks.append(ipaddress.IPv4Network(network_str))
        except ipaddress.AddressValueError:
            continue

    return networks


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address as string
    """
    # Check for forwarded headers (proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fallback to direct connection
    if hasattr(request.client, 'host'):
        return request.client.host

    return "unknown"


def is_ip_allowed(ip_address: str) -> bool:
    """Check if an IP address is in the allowed networks.

    Args:
        ip_address: IP address to check

    Returns:
        True if IP is allowed, False otherwise
    """
    # If local network requirement is disabled, allow all
    if not REQUIRE_LOCAL_NETWORK and not CUSTOM_NETWORKS:
        return True

    try:
        ip = ipaddress.IPv4Address(ip_address)
        allowed_networks = get_allowed_networks()

        # Check if IP is in any allowed network
        for network in allowed_networks:
            if ip in network:
                return True

        return False

    except (ipaddress.AddressValueError, ValueError):
        # Invalid IP address format
        return False


def is_localhost(ip_address: str) -> bool:
    """Check if IP address is localhost.

    Args:
        ip_address: IP address to check

    Returns:
        True if IP is localhost, False otherwise
    """
    try:
        ip = ipaddress.IPv4Address(ip_address)
        return ip.is_loopback
    except (ipaddress.AddressValueError, ValueError):
        return ip_address.lower() in ["localhost", "::1"]


def is_private_network(ip_address: str) -> bool:
    """Check if IP address is in a private network range.

    Args:
        ip_address: IP address to check

    Returns:
        True if IP is private, False otherwise
    """
    try:
        ip = ipaddress.IPv4Address(ip_address)
        return ip.is_private
    except (ipaddress.AddressValueError, ValueError):
        return False


def get_network_info(ip_address: str) -> dict:
    """Get detailed network information for an IP address.

    Args:
        ip_address: IP address to analyze

    Returns:
        Dictionary with network information
    """
    info = {
        "ip": ip_address,
        "allowed": is_ip_allowed(ip_address),
        "localhost": is_localhost(ip_address),
        "private": is_private_network(ip_address),
        "network_type": "unknown"
    }

    try:
        ip = ipaddress.IPv4Address(ip_address)

        if ip.is_loopback:
            info["network_type"] = "loopback"
        elif ip.is_private:
            info["network_type"] = "private"
        elif ip.is_global:
            info["network_type"] = "public"
        elif ip.is_link_local:
            info["network_type"] = "link_local"
        elif ip.is_multicast:
            info["network_type"] = "multicast"

        # Check which allowed network it belongs to
        if info["allowed"]:
            allowed_networks = get_allowed_networks()
            for network in allowed_networks:
                if ip in network:
                    info["matched_network"] = str(network)
                    break

    except (ipaddress.AddressValueError, ValueError):
        info["network_type"] = "invalid"

    return info


def validate_network_config() -> dict:
    """Validate the current network security configuration.

    Returns:
        Dictionary with validation results
    """
    allowed_networks = get_allowed_networks()

    return {
        "require_local_network": REQUIRE_LOCAL_NETWORK,
        "allowed_networks": [str(net) for net in allowed_networks],
        "custom_networks": CUSTOM_NETWORKS,
        "total_networks": len(allowed_networks),
        "security_level": "high" if REQUIRE_LOCAL_NETWORK else "disabled"
    }