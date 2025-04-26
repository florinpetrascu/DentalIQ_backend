
from sqlalchemy import  Column, Integer, String
from sqlalchemy.orm import relationship

from database.base import Base


class Patient(Base):
    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)

    teeths = relationship("Teeth", backref="patient", cascade="all, delete-orphan")

    def addTooth(self,tooth):
        self.teeths.append(tooth)
    def __str__(self):
        return f"Patient {self.id}: {self.first_name} {self.last_name}, Phone: {self.phone_number}"

    def __repr__(self):
        return f"<Patient(id={self.id}, first_name='{self.first_name}', last_name='{self.last_name}', phone_number='{self.phone_number}')>"