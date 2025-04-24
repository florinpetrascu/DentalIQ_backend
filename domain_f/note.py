
from database.base import Base


class Note:

    def __init__(self, id, date, text):
        self.id = id
        self.date = date
        self.text = text

