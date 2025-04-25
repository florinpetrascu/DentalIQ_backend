
from sqlalchemy import  Column, Integer, String ,ForeignKey
from database.base import Base



class Radiography(Base):
    __tablename__ = 'radiographs'

    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)  # Cheie străină
