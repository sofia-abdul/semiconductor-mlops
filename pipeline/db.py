from sqlalchemy import create_engine
from pipeline.config import DB_URL


def get_engine():
     """Create and return a SQLAlchemy database engine."""
    return create_engine(DB_URL)