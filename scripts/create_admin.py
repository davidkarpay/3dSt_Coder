#!/usr/bin/env python3
"""Create admin user for 3dSt Platform authentication system."""

import asyncio
import os
import sys
import getpass
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auth.database import auth_db
from auth.models import UserCreate, UserRole
from auth.security import validate_password_strength


async def create_admin_user() -> None:
    """Create admin user interactively."""
    print("3dSt Platform Admin User Creation")
    print("=" * 40)

    try:
        # Initialize database
        await auth_db.initialize()

        # Check if admin already exists
        users = await auth_db.list_users()
        admin_exists = any(user.role == UserRole.ADMIN for user in users)

        if admin_exists:
            print("⚠️  Admin user already exists!")
            response = input("Do you want to create another admin user? (y/N): ").lower()
            if response != 'y':
                print("Aborted.")
                return

        # Get username
        while True:
            username = input("Enter admin username (3-50 characters): ").strip()
            if not username:
                print("❌ Username cannot be empty")
                continue
            if len(username) < 3 or len(username) > 50:
                print("❌ Username must be 3-50 characters long")
                continue

            # Check if username exists
            existing_user = await auth_db.get_user_by_username(username)
            if existing_user:
                print(f"❌ Username '{username}' already exists")
                continue

            break

        # Get email (optional)
        email = input("Enter admin email (optional): ").strip()
        if email and "@" not in email:
            print("⚠️  Invalid email format, continuing without email")
            email = None

        # Get password
        while True:
            password = getpass.getpass("Enter admin password: ")
            if not password:
                print("❌ Password cannot be empty")
                continue

            # Validate password strength
            validation = validate_password_strength(password)
            if not validation["valid"]:
                print("❌ Password does not meet requirements:")
                for error in validation["errors"]:
                    print(f"   • {error}")
                continue

            # Confirm password
            confirm_password = getpass.getpass("Confirm admin password: ")
            if password != confirm_password:
                print("❌ Passwords do not match")
                continue

            print(f"✅ Password strength: {validation['strength']}")
            break

        # Create user data
        user_data = UserCreate(
            username=username,
            email=email if email else None,
            password=password,
            role=UserRole.ADMIN
        )

        # Confirm creation
        print("\nAdmin User Details:")
        print(f"Username: {username}")
        print(f"Email: {email or 'Not provided'}")
        print(f"Role: Admin")

        response = input("\nCreate this admin user? (y/N): ").lower()
        if response != 'y':
            print("Aborted.")
            return

        # Create user
        user = await auth_db.create_user(user_data)

        print(f"\n✅ Admin user '{username}' created successfully!")
        print(f"User ID: {user.id}")
        print(f"Created at: {user.created_at}")

        # Security reminder
        print("\n🔐 Security Reminders:")
        print("• Store the admin credentials securely")
        print("• Consider changing the password after first login")
        print("• Only access the system from allowed networks/VPN")
        print("• Monitor access logs regularly")

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n❌ Error creating admin user: {e}")
        sys.exit(1)
    finally:
        await auth_db.close()


async def list_users() -> None:
    """List existing users."""
    try:
        await auth_db.initialize()
        users = await auth_db.list_users()

        if not users:
            print("No users found.")
            return

        print(f"\nExisting Users ({len(users)}):")
        print("-" * 60)
        for user in users:
            status = "✅ Active" if user.active else "❌ Inactive"
            role_icon = "👑" if user.role == UserRole.ADMIN else "👤"
            print(f"{role_icon} {user.username:<20} {user.role:<10} {status}")

    except Exception as e:
        print(f"❌ Error listing users: {e}")
    finally:
        await auth_db.close()


def print_usage():
    """Print script usage information."""
    print("Usage:")
    print(f"  {sys.argv[0]} create    - Create a new admin user")
    print(f"  {sys.argv[0]} list      - List existing users")
    print(f"  {sys.argv[0]} --help    - Show this help message")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command in ["--help", "-h", "help"]:
        print_usage()
        return
    elif command == "create":
        asyncio.run(create_admin_user())
    elif command == "list":
        asyncio.run(list_users())
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    main()