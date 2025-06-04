


from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import declarative_base

from database.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nume = Column(String)
    prenume = Column(String)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, nullable=True)
    activ = Column(Boolean, default=True)