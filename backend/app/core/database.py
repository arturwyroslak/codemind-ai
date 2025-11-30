from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://codemind:codemind@localhost:5432/codemind"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AuditLog(Base):
    """Audit log for tracking all operations."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String(64), unique=True, index=True)
    user_id = Column(String(64), index=True)
    operation = Column(String(128), index=True)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(32), default="success")
    error_message = Column(Text, nullable=True)

class AnalysisResult(Base):
    """Store analysis results for caching and history."""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    code_hash = Column(String(64), index=True)
    analysis_type = Column(String(32))
    language = Column(String(32))
    result = Column(JSON)
    overall_score = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String(64), index=True)

class APIKey(Base):
    """API keys for authentication."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), unique=True, index=True)
    user_id = Column(String(64), index=True)
    name = Column(String(128))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Integer, default=1)
    rate_limit = Column(Integer, default=1000)  # requests per hour

def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)