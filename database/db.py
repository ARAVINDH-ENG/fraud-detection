import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fraud.db")

if "sqlite" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
