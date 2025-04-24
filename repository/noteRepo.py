from domain.note import Note


class NotesRepo:

    def __init__(self, session):
        self.session = session

    # CRUD pentru Notes
    def create_notes(self, text, date, teeth_id):
        new_note = Note(None, text, date, teeth_id)
        self.session.add(new_note)
        self.session.commit()

    def get_notes_by_id(self, note_id):
        return self.session.query(Note).filter(Note.id == note_id).first()

    def update_notes(self, note_id, text, date, teeth_id):
        note = self.get_notes_by_id(note_id)
        if note:
            note.text = text
            note.date = date
            note.teeth_id = teeth_id
            self.session.commit()

    def delete_notes(self, note_id):
        note = self.get_notes_by_id(note_id)
        if note:
            self.session.delete(note)
            self.session.commit()
