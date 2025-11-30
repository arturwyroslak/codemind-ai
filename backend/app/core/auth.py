from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import hashlib
import secrets
from datetime import datetime
from typing import Optional

from .database import get_db, APIKey

security = HTTPBearer()

class AuthService:
    """Authentication and authorization service."""
    
    @staticmethod
    def generate_api_key(user_id: str, name: str = "default") -> str:
        """Generate a new API key."""
        raw_key = secrets.token_urlsafe(48)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        return f"cm_{raw_key}"
    
    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    @staticmethod
    async def verify_token(
        credentials: HTTPAuthorizationCredentials = Security(security),
        db: Session = Depends(get_db)
    ) -> dict:
        """Verify API token and return user info."""
        token = credentials.credentials
        
        # For development: allow test token
        if token == "test-token-dev-only":
            return {
                "user_id": "dev_user",
                "rate_limit": 10000,
                "is_admin": True
            }
        
        # Hash the token
        token_hash = AuthService.hash_key(token)
        
        # Query database
        api_key = db.query(APIKey).filter(
            APIKey.key == token_hash,
            APIKey.is_active == 1
        ).first()
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired API key"
            )
        
        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=401,
                detail="API key has expired"
            )
        
        return {
            "user_id": api_key.user_id,
            "rate_limit": api_key.rate_limit,
            "is_admin": False
        }

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """Dependency to get current authenticated user."""
    # Simplified for now - just extract user_id from token
    token = credentials.credentials
    
    # For JWT tokens, decode here
    # For now, use simple token prefix
    if token.startswith("cm_"):
        return {"user_id": token[:16], "rate_limit": 1000}
    
    return {"user_id": token[:8], "rate_limit": 100}