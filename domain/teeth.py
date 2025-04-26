
from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from database.base import Base


class Teeth(Base):
    __tablename__ = 'teeth'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    polygon = Column(Text, nullable=True)
    patient_id = Column(Integer, ForeignKey('patients.id', ondelete="CASCADE"), nullable=True)  # Cheie străină

    issues = relationship("Issue", back_populates="tooth", cascade="all, delete-orphan")
    #notes = relationship("Note", back_populates="tooth", cascade="all, delete-orphan")

    def add_issue(self,issue):
        self.issues.append(issue)

    def __repr__(self):
        return f"<Teeth( name='{self.name}', polygon='{self.polygon}' , issues='{self.issues}')>"

    def __str__(self):
        return f"Teeth(name='{self.name}', polygon='{self.polygon}, issues='{self.issues}')"