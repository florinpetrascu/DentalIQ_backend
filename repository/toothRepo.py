from domain.teeth import Teeth


class TeethRepo:

    def __init__(self, session):
        self.session = session

    # CRUD pentru Teeth
    def create_teeth(self, name, polygon, patient_id):
        new_teeth = Teeth(None, name, polygon, patient_id)
        self.session.add(new_teeth)
        self.session.commit()

    def find_by_patient_id(self, patient_id):
        return self.session.query(Teeth).filter(Teeth.patient_id == patient_id)

    def get_teeth_by_id(self, teeth_id):
        return self.session.query(Teeth).filter(Teeth.id == teeth_id).first()

    def update_teeth(self, teeth_id, name, polygon, patient_id):
        teeth = self.get_teeth_by_id(teeth_id)
        if teeth:
            teeth.name = name
            teeth.polygon = polygon
            teeth.patient_id = patient_id
            self.session.commit()

    def delete_by_patient_id(self, patient_id):

        self.session.query(Teeth).filter(Teeth.patient_id == patient_id).delete()

    def save(self,tooth):
        self.session.add(tooth)
        self.session.commit()