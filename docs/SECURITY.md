# 🔐 Security Guide

Comprehensive security documentation for 3dSt_Coder, designed for law firms and sensitive environments.

## Table of Contents

- [Security Overview](#security-overview)
- [Authentication System](#authentication-system)
- [Network Security](#network-security)
- [Data Protection](#data-protection)
- [Access Control](#access-control)
- [Session Management](#session-management)
- [Tool Security](#tool-security)
- [Compliance Considerations](#compliance-considerations)
- [Security Best Practices](#security-best-practices)
- [Incident Response](#incident-response)

## Security Overview

3dSt_Coder implements enterprise-grade security features specifically designed for law firms and organizations handling sensitive information:

### Core Security Principles

- **Local-First**: All processing happens on your machine - no data transmission to external servers
- **Zero-Trust Network**: Default deny with explicit allow for local networks only
- **Defense in Depth**: Multiple security layers across authentication, network, application, and data
- **Principle of Least Privilege**: Users and processes only have minimum required permissions
- **Data Isolation**: Per-user conversation history and session separation

### Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Network Layer Security                    │
├─────────────────────────────────────────────────────────────┤
│ IP Validation → Local Networks Only (192.168.x.x, 10.x.x.x) │
│ VPN Support → Custom Network Range Configuration            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Authentication Layer Security                │
├─────────────────────────────────────────────────────────────┤
│ JWT Tokens → Signed with Secret Key, 8-hour Expiration     │
│ Password Security → bcrypt Hashing, Strength Validation     │
│ Session Management → Token Tracking, Automatic Cleanup      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Application Layer Security                   │
├─────────────────────────────────────────────────────────────┤
│ Role-Based Access → Admin/User Permissions                  │
│ Input Validation → Pydantic Models, SQL Injection Prevention│
│ Path Protection → File Operations Restricted to Project Dir │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer Security                       │
├─────────────────────────────────────────────────────────────┤
│ User Isolation → Per-User Conversation History              │
│ Local Storage → SQLite Database, No Cloud Dependencies      │
│ Command Sandboxing → Restricted Shell Operations            │
└─────────────────────────────────────────────────────────────┘
```

## Authentication System

### JWT Token Security

**Token Generation:**
```python
# Tokens are signed with HS256 algorithm
# Default expiration: 8 hours (configurable)
# Include user ID, username, and role claims
```

**Security Features:**
- **Cryptographic Signing**: Tokens signed with secret key to prevent tampering
- **Expiration Control**: Configurable token lifetime (default 8 hours)
- **Automatic Refresh**: Tokens refreshed during active sessions
- **Secure Storage**: Client-side storage in httpOnly cookies (planned) or localStorage

**Configuration:**
```bash
# Production Security Settings
AUTH_SECRET_KEY=your-unique-256-bit-secret-key-here
AUTH_TOKEN_EXPIRE_MINUTES=480  # 8 hours
```

### Password Security

**Requirements:**
- **Minimum Length**: 8 characters
- **Complexity**: Must include uppercase, lowercase, numbers, and special characters
- **Common Password Protection**: Blocks common passwords like "password123"
- **Strength Validation**: Real-time feedback during password creation

**Hashing:**
- **Algorithm**: bcrypt with salt rounds (cost factor 12)
- **Storage**: Only hashed passwords stored, plaintext never persisted
- **Verification**: Constant-time comparison to prevent timing attacks

**Example Strong Passwords:**
```
✅ SecureP@ssw0rd2024!
✅ L@wF1rm$3cur1ty#
✅ C0d3r@ccess2024$
❌ password123
❌ 12345678
❌ admin
```

### User Management

**Admin Controls:**
- Create new user accounts
- Deactivate user accounts (prevents login)
- View user list and session activity
- Cleanup expired sessions
- Cannot deactivate their own account

**User Registration Process:**
1. Admin creates user account with username/email/password
2. Password strength validation enforced
3. User receives credentials to login
4. First login creates active session
5. User can change password after login (planned feature)

## Network Security

### IP Address Validation

**Default Allowed Networks:**
```
127.0.0.0/8     - Localhost (always allowed)
10.0.0.0/8      - Private Class A networks
172.16.0.0/12   - Private Class B networks
192.168.0.0/16  - Private Class C networks
```

**Custom Network Configuration:**
```bash
# Allow specific VPN subnets
AUTH_ALLOWED_NETWORKS=10.0.0.0/8,192.168.0.0/16,172.20.0.0/16

# Add corporate network ranges
AUTH_ALLOWED_NETWORKS=10.0.0.0/8,192.168.0.0/16,203.0.113.0/24
```

### Network Access Control Process

1. **Client Connection**: User attempts to access any endpoint
2. **IP Extraction**: System extracts client IP address
   - Supports X-Forwarded-For header for proxy environments
   - Handles X-Real-IP for reverse proxy setups
3. **Network Validation**: IP checked against allowed networks
4. **Access Decision**: Allow or deny based on network policy

### Firewall Recommendations

**Windows Firewall:**
```powershell
# Allow 3dSt_Coder only on private networks
New-NetFirewallRule -DisplayName "3dSt_Coder Local" -Direction Inbound -Protocol TCP -LocalPort 8000 -Profile Private -Action Allow

# Block public network access
New-NetFirewallRule -DisplayName "3dSt_Coder Block Public" -Direction Inbound -Protocol TCP -LocalPort 8000 -Profile Public -Action Block
```

**Linux iptables:**
```bash
# Allow local network access only
iptables -A INPUT -p tcp --dport 8000 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

## Data Protection

### Local-First Architecture

**No External Dependencies:**
- All AI processing happens locally using Ollama/vLLM
- No API calls to external LLM services (unless explicitly configured)
- Conversation history stored locally in SQLite
- User credentials stored locally with encryption

**Data Storage Locations:**
```
data/conversations/          # Conversation history (SQLite)
data/auth.db                # User accounts and sessions (SQLite)
logs/                       # Application logs (local files)
```

### User Data Isolation

**Conversation Isolation:**
- Each user has separate conversation history
- Users cannot access other users' conversations
- Database queries scoped by user ID
- Memory management per-user

**Session Isolation:**
- JWT tokens unique per user session
- Session tracking prevents token reuse across users
- Automatic session cleanup on logout
- Expired session cleanup (admin function)

### File System Security

**Path Traversal Protection:**
```python
# All file operations validated against project directory
# Prevents access to files outside project scope
# Uses os.path.commonpath for validation
```

**Restricted File Operations:**
- Read operations: Limited to project directory tree
- Write operations: Limited to project directory tree
- No access to system files (/etc, /windows, etc.)
- No access to user home directories outside project

## Access Control

### Role-Based Access Control (RBAC)

**Admin Role:**
- Full system access
- User management capabilities
- Session management and cleanup
- System configuration access
- Cannot deactivate own account

**User Role:**
- Chat interface access
- Own conversation history
- Own session management
- Cannot access admin functions
- Cannot view other users' data

### API Endpoint Security

**Public Endpoints (No Authentication Required):**
```
GET  /auth/status     # Network and auth status
POST /auth/login      # User authentication
GET  /api/v1/health   # Basic health check
```

**Authenticated Endpoints (Require Valid JWT):**
```
POST /api/v1/chat           # Chat interactions
POST /api/v1/chat/complete  # Synchronous chat
GET  /api/v1/conversations  # User's conversation history
GET  /api/v1/tools          # Available tools
POST /auth/logout           # Session termination
GET  /auth/me               # Current user info
```

**Admin-Only Endpoints:**
```
POST /auth/register           # Create new users
GET  /auth/users              # List all users
POST /auth/users/{id}/deactivate  # Deactivate users
POST /auth/cleanup-sessions   # Clean expired sessions
```

### Permission Matrix

| Action | User | Admin |
|--------|------|-------|
| Login | ✅ | ✅ |
| Chat | ✅ | ✅ |
| View own conversations | ✅ | ✅ |
| View all conversations | ❌ | ✅ |
| Create users | ❌ | ✅ |
| Deactivate users | ❌ | ✅ |
| System administration | ❌ | ✅ |

## Session Management

### Session Lifecycle

1. **Session Creation**: On successful login
   - JWT token generated with expiration
   - Session record stored with metadata
   - Client receives token for future requests

2. **Session Validation**: On each request
   - Token signature verification
   - Expiration time checking
   - Session record validation
   - User active status checking

3. **Session Termination**:
   - **Manual Logout**: User clicks logout
   - **Token Expiration**: Automatic after 8 hours
   - **Admin Deactivation**: Session immediately invalidated
   - **Server Restart**: All sessions cleared

### Session Security Features

**Session Tracking:**
```sql
-- Session metadata stored for audit trail
CREATE TABLE user_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT
);
```

**Automatic Cleanup:**
- Expired sessions automatically deleted
- Admin can manually trigger cleanup
- Database maintenance prevents session table growth
- Audit trail preserved for active sessions only

**Security Controls:**
- One session per token (no token sharing)
- IP address logging for audit trail
- User agent tracking for anomaly detection
- Configurable session timeout

## Tool Security

### Sandboxed Execution

**Command Filtering:**
```python
# Dangerous commands blocked by default
BLOCKED_COMMANDS = [
    'rm -rf', 'del /f', 'format', 'fdisk',
    'chmod 777', 'sudo', 'su', 'passwd',
    'curl', 'wget', 'nc', 'netcat'
]
```

**Execution Environment:**
- Commands run in controlled subprocess
- Working directory restricted to project
- Environment variables sanitized
- Resource limits applied (memory, CPU, time)

### File Operation Security

**Path Validation:**
```python
def validate_path(requested_path: str, project_root: str) -> bool:
    # Resolve symlinks and relative paths
    # Check if path is within project directory
    # Prevent access to system files
    return os.path.commonpath([requested_path, project_root]) == project_root
```

**Allowed Operations:**
- Read files within project directory
- Write files within project directory
- Create directories within project scope
- Git operations on project repository

**Blocked Operations:**
- Access to system directories (/etc, /windows)
- Access to user home directories
- Network file operations
- Binary execution outside project

### Git Security

**Safe Git Operations:**
- Status checking (read-only)
- Diff viewing (read-only)
- Commit creation (project scope only)
- Branch operations (local repository only)

**Blocked Git Operations:**
- Remote operations (push, pull, fetch)
- Repository cloning
- Submodule operations
- Hook execution

## Compliance Considerations

### Legal Industry Requirements

**Attorney-Client Privilege:**
- All data processing happens locally
- No transmission to external services
- Conversation history encrypted at rest
- User access controls prevent data leakage

**Data Retention:**
- Configurable conversation retention periods
- Secure deletion capabilities
- Audit trail for data access
- Compliance reporting features (planned)

**Access Auditing:**
- User login/logout events logged
- API access patterns tracked
- Failed authentication attempts recorded
- Session anomaly detection

### GDPR Compliance Features

**Data Minimization:**
- Only required user data collected
- Optional email field for user accounts
- Conversation history scoped per user
- No tracking or analytics data

**Right to Erasure:**
- Admin can deactivate user accounts
- Conversation history tied to user accounts
- Secure deletion of user data (planned)
- Data export capabilities (planned)

**Data Processing Transparency:**
- Local processing clearly documented
- No third-party data sharing
- User consent for data collection
- Privacy policy for usage patterns

## Security Best Practices

### Deployment Security

**Environment Hardening:**
```bash
# Use strong secret key in production
AUTH_SECRET_KEY=$(openssl rand -hex 32)

# Restrict network access
AUTH_REQUIRE_LOCAL_NETWORK=true
AUTH_ALLOWED_NETWORKS=192.168.1.0/24

# Set appropriate token expiration
AUTH_TOKEN_EXPIRE_MINUTES=240  # 4 hours for high security
```

**System Security:**
- Run as non-root user
- Use dedicated service account
- Apply OS security patches regularly
- Monitor system resource usage

### Password Management

**Admin Account Security:**
- Use unique, complex passwords
- Regular password rotation (quarterly)
- Multi-factor authentication (planned)
- Account lockout after failed attempts (planned)

**User Account Guidelines:**
- Enforce strong password policy
- Provide password strength feedback
- Regular password audits
- Account deactivation for inactive users

### Monitoring and Alerting

**Security Events to Monitor:**
```
- Multiple failed login attempts
- Login from new IP addresses
- Session anomalies (unusual duration/activity)
- Admin account usage
- API rate limiting triggers
- System resource exhaustion
```

**Log Analysis:**
```bash
# Monitor authentication failures
grep "Failed login attempt" logs/app.log

# Track admin activities
grep "admin.*created\|deactivated" logs/app.log

# Monitor network access denials
grep "Network access denied" logs/app.log
```

### Backup and Recovery

**Secure Backup Procedures:**
1. **Database Backup**: Regular SQLite database backups
2. **Configuration Backup**: Environment variables and settings
3. **Conversation History**: User data backup with encryption
4. **Recovery Testing**: Regular restore procedure validation

**Backup Security:**
- Encrypt backup files
- Store backups securely (encrypted storage)
- Regular backup integrity verification
- Offsite backup storage (air-gapped)

## Incident Response

### Security Incident Classification

**Level 1 - Low Risk:**
- Single failed login attempt
- Normal session timeout
- Expected network access denial

**Level 2 - Medium Risk:**
- Multiple failed login attempts from same IP
- Session anomaly detected
- Unusual API usage patterns

**Level 3 - High Risk:**
- Potential brute force attack
- Unauthorized access attempt
- System compromise indicators
- Data access anomalies

### Response Procedures

**Immediate Actions:**
1. **Assess Threat Level**: Determine incident severity
2. **Preserve Evidence**: Backup logs and system state
3. **Contain Incident**: Block malicious IPs, disable accounts
4. **Document Timeline**: Record all actions and observations

**Investigation Steps:**
1. **Log Analysis**: Review authentication and access logs
2. **Network Forensics**: Analyze network traffic patterns
3. **System Integrity**: Check for unauthorized changes
4. **User Account Review**: Audit user access and permissions

**Recovery Actions:**
1. **System Restoration**: Restore from clean backups if needed
2. **Password Reset**: Force password reset for affected accounts
3. **Session Cleanup**: Invalidate all active sessions
4. **Security Patch**: Apply any necessary security updates

### Emergency Contacts

**Internal Escalation:**
- IT Security Team
- Legal Counsel (for legal industry clients)
- System Administrator
- Management Team

**External Resources:**
- Law Enforcement (for criminal activity)
- Cybersecurity Consultants
- Legal Industry Security Experts
- Incident Response Services

---

## 🆘 Additional Resources

- **[Setup Guide](SETUP.md)** - Security configuration options
- **[User Guide](USER_GUIDE.md)** - Authentication and security features
- **[Troubleshooting](TROUBLESHOOTING.md)** - Security-related issues
- **[Quick Start](QUICK_START.md)** - Secure installation guide

**For security questions or incident reporting, consult your organization's IT security policies and procedures.**