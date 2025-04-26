


from sqlalchemy import Column, Integer,ForeignKey, Text,TIMESTAMP

from database.base import Base
class Note(Base):
    __tablename__ = 'notes'

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    date = Column(TIMESTAMP, nullable=False)
    teeth_id = Column(Integer, ForeignKey('teeth.id'))

    #tooth = relationship("Teeth", back_populates="notes")
    ##mere treaba aici