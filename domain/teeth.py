
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from database.base import Base


class Teeth(Base):
    __tablename__ = 'teeth'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    polygon = Column(Text, nullable=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)  # Cheie străină

    patient = relationship("Patient", back_populates="teeths")
    issues = relationship("Issue", back_populates="tooth")
    notes = relationship("Note", back_populates="tooth")

    def addIssue(self,issue):
        self.issues.append(issue)

    def __repr__(self):
        return f"<Teeth( name='{self.name}', polygon='{self.polygon}' , issues='{self.issues}')>"

    def __str__(self):
        return f"Teeth(name='{self.name}', polygon='{self.polygon}, issues='{self.issues}')"