

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table
from sqlalchemy.orm import relationship, sessionmaker, declarative_base


from database.base import Base
class Issue(Base):
    __tablename__ = 'issues'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    tooth_id = Column(Integer, ForeignKey('teeth.id'))

    tooth = relationship("Teeth", back_populates="issues")
