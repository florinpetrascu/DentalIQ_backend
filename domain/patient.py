
from sqlalchemy import  Column, Integer, String
from sqlalchemy.orm import relationship

from database.base import Base


class Patient(Base):
    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True)
    firstName = Column(String, nullable=False)
    lastName = Column(String, nullable=False)
    phoneNumber = Column(String, nullable=False)
    image = Column(String, nullable=True)

    teeths = relationship("Teeth", back_populates="patient")
