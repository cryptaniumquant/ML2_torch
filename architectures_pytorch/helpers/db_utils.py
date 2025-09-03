from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .constants import DATABASE_URL
import pandas as pd

def get_db_engine():
    """Create and return a database engine"""
    return create_engine(DATABASE_URL)

def get_db_session():
    """Create and return a database session"""
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def execute_query(query):
    """Execute a SQL query and return results as a pandas DataFrame"""
    engine = get_db_engine()
    return pd.read_sql(query, engine)