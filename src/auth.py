"""
KAIROS - Authentication & Role-Based Access Control Module
Provides secure salted password hashing, role management, and session helpers.
Roles: 'student', 'admin'
"""

import hashlib
import hmac
import os
from typing import Optional, Dict, Any, Tuple


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password with SHA-256 and salt.
    Returns (salt, password_hash).
    """
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return salt, hashed


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verifies a password against stored salt and hash in constant time."""
    _, calculated_hash = hash_password(password, salt=salt)
    return hmac.compare_digest(calculated_hash, expected_hash)
