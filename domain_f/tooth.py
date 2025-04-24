
from database.base import Base
class Tooth:

    def __init__(self, id, name, polygon):
        self.id = id
        self.name = name
        self.polygon = polygon
        self.issues = []
        self.notes = []

    # SETTERS
    def set_name(self, name):
        self.name = name

    def set_polygon(self, polygon):
        self.polygon = polygon

    def set_issues(self, issues):
        self.issues = issues

    def set_notes(self, notes):
        self.notes = notes

    def add_issue(self, issue):
        self.issues.append(issue)

    def add_note(self, note):
        self.notes.append(note)

