"""Database operations for authentication."""

import logging
import aiosqlite
import os
from typing import Optional, List
from datetime import datetime, timedelta
from .models import User, UserInDB, UserCreate, UserResponse, UserRole, Session
from .security import get_password_hash, hash_token

logger = logging.getLogger(__name__)


class AuthDatabase:
    """Database manager for authentication operations."""

    def __init__(self, db_path: str = "data/auth.db"):
        """Initialize authentication database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection = None

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        logger.info(f"Authentication database initialized: {self.db_path}")

    async def _create_tables(self) -> None:
        """Create authentication tables."""
        if not self._connection:
            return

        # Users table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                active BOOLEAN NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sessions table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)
        """)

        await self._connection.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user.

        Args:
            user_data: User creation data

        Returns:
            Created user data

        Raises:
            ValueError: If username already exists
        """
        if not self._connection:
            raise RuntimeError("Database not initialized")

        # Check if username exists
        cursor = await self._connection.execute(
            "SELECT id FROM users WHERE username = ?",
            (user_data.username,)
        )
        if await cursor.fetchone():
            raise ValueError(f"Username '{user_data.username}' already exists")

        # Hash password
        password_hash = get_password_hash(user_data.password)

        # Insert user
        cursor = await self._connection.execute("""
            INSERT INTO users (username, email, password_hash, role, active)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_data.username,
            user_data.email,
            password_hash,
            user_data.role.value,
            True
        ))

        user_id = cursor.lastrowid
        await self._connection.commit()

        # Return created user
        return await self.get_user_by_id(user_id)

    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username.

        Args:
            username: Username to search for

        Returns:
            User data if found, None otherwise
        """
        if not self._connection:
            return None

        cursor = await self._connection.execute("""
            SELECT id, username, email, password_hash, role, active, created_at
            FROM users WHERE username = ? AND active = 1
        """, (username,))

        row = await cursor.fetchone()
        if not row:
            return None

        return UserInDB(
            id=row[0],
            username=row[1],
            email=row[2],
            password_hash=row[3],
            role=UserRole(row[4]),
            active=bool(row[5]),
            created_at=datetime.fromisoformat(row[6]) if row[6] else None
        )

    async def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID.

        Args:
            user_id: User ID to search for

        Returns:
            User data if found, None otherwise
        """
        if not self._connection:
            return None

        cursor = await self._connection.execute("""
            SELECT id, username, email, role, active, created_at
            FROM users WHERE id = ? AND active = 1
        """, (user_id,))

        row = await cursor.fetchone()
        if not row:
            return None

        return UserResponse(
            id=row[0],
            username=row[1],
            email=row[2],
            role=UserRole(row[3]),
            active=bool(row[4]),
            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now()
        )

    async def list_users(self) -> List[UserResponse]:
        """List all active users.

        Returns:
            List of user data
        """
        if not self._connection:
            return []

        cursor = await self._connection.execute("""
            SELECT id, username, email, role, active, created_at
            FROM users WHERE active = 1
            ORDER BY created_at DESC
        """)

        users = []
        async for row in cursor:
            users.append(UserResponse(
                id=row[0],
                username=row[1],
                email=row[2],
                role=UserRole(row[3]),
                active=bool(row[4]),
                created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now()
            ))

        return users

    async def create_session(self, user_id: int, token: str, expires_at: datetime,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> int:
        """Create a new session.

        Args:
            user_id: User ID
            token: JWT token
            expires_at: Session expiration time
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session ID
        """
        if not self._connection:
            raise RuntimeError("Database not initialized")

        token_hash_value = hash_token(token)

        cursor = await self._connection.execute("""
            INSERT INTO sessions (user_id, token_hash, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, token_hash_value, expires_at.isoformat(), ip_address, user_agent))

        session_id = cursor.lastrowid
        await self._connection.commit()
        return session_id

    async def get_session(self, token: str) -> Optional[Session]:
        """Get session by token.

        Args:
            token: JWT token

        Returns:
            Session data if found and valid, None otherwise
        """
        if not self._connection:
            return None

        token_hash_value = hash_token(token)

        cursor = await self._connection.execute("""
            SELECT id, user_id, token_hash, expires_at, created_at, ip_address, user_agent
            FROM sessions
            WHERE token_hash = ? AND expires_at > datetime('now')
        """, (token_hash_value,))

        row = await cursor.fetchone()
        if not row:
            return None

        return Session(
            id=row[0],
            user_id=row[1],
            token_hash=row[2],
            expires_at=datetime.fromisoformat(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            ip_address=row[5],
            user_agent=row[6]
        )

    async def delete_session(self, token: str) -> bool:
        """Delete a session.

        Args:
            token: JWT token

        Returns:
            True if session was deleted, False otherwise
        """
        if not self._connection:
            return False

        token_hash_value = hash_token(token)

        cursor = await self._connection.execute("""
            DELETE FROM sessions WHERE token_hash = ?
        """, (token_hash_value,))

        await self._connection.commit()
        return cursor.rowcount > 0

    async def delete_expired_sessions(self) -> int:
        """Delete all expired sessions.

        Returns:
            Number of deleted sessions
        """
        if not self._connection:
            return 0

        cursor = await self._connection.execute("""
            DELETE FROM sessions WHERE expires_at <= datetime('now')
        """)

        await self._connection.commit()
        return cursor.rowcount

    async def delete_user_sessions(self, user_id: int) -> int:
        """Delete all sessions for a user.

        Args:
            user_id: User ID

        Returns:
            Number of deleted sessions
        """
        if not self._connection:
            return 0

        cursor = await self._connection.execute("""
            DELETE FROM sessions WHERE user_id = ?
        """, (user_id,))

        await self._connection.commit()
        return cursor.rowcount

    async def update_user_active_status(self, user_id: int, active: bool) -> bool:
        """Update user active status.

        Args:
            user_id: User ID
            active: New active status

        Returns:
            True if user was updated, False otherwise
        """
        if not self._connection:
            return False

        cursor = await self._connection.execute("""
            UPDATE users SET active = ? WHERE id = ?
        """, (active, user_id))

        await self._connection.commit()
        return cursor.rowcount > 0


# Global database instance
auth_db = AuthDatabase()