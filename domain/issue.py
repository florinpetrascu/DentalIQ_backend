

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship


from database.base import Base
class Issue(Base):
    __tablename__ = 'issues'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    teeth_id = Column(Integer, ForeignKey('teeth.id', ondelete="CASCADE"), nullable=True)

    tooth = relationship("Teeth", back_populates="issues")

    def __repr__(self):
        return f"<Issue( id='{self.id}', name='{self.name}', teeth_id='{self.teeth_id}')>"

    def __str__(self):
        return f"Issue( id='{self.id}', name='{self.name}', teeth_id='{self.teeth_id}')"