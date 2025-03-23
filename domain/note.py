


from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from database.base import Base
class Note(Base):
    __tablename__ = 'notes'

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    tooth_id = Column(Integer, ForeignKey('teeth.id'))

    tooth = relationship("Teeth", back_populates="notes")
    ##mere treaba aici