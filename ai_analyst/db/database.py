import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DATABASE URL
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not configured."
    )


# ============================================================
# SQLALCHEMY BASE
# ============================================================
#
# IMPORTANT:
# Base must be created BEFORE importing models.
# This prevents the circular import:
#
# database.py -> models.py -> database.py
#
# ============================================================

Base = declarative_base()


# ============================================================
# DATABASE ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)


# ============================================================
# DATABASE SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():
    """
    Import models only when database initialization happens.

    This avoids circular imports during application startup.
    """

    from . import models

    Base.metadata.create_all(
        bind=engine
    )

    print(
        "✅ PostgreSQL tables initialized successfully."
    )