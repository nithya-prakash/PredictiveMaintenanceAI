import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.core.config import settings

log = logging.getLogger(__name__)

engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> bool:
    """Creates tables if the database is reachable. Called on API start-up (not at
    import time, so importing the app never requires a running database)."""
    from backend.models import machine  # noqa: F401  (registers the ORM models)
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except SQLAlchemyError as e:
        log.warning("Database not available, predictions will not be persisted: %s", e.__class__.__name__)
        return False


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
