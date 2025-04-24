
from domain.patient import Patient

class PatientRepo:


    def __init__(self,session):
            self.session = session

    # CRUD pentru Patient
    def create_patient(self,first_name, last_name, phone_number):
        new_patient = Patient(None, first_name, last_name, phone_number)
        self.session.add(new_patient)
        self.session.commit()

    def get_patient_by_id(self, patient_id):
        return self.session.query(Patient).filter(Patient.id == patient_id).first()

    def update_patient(self, patient_id, first_name, last_name, phone_number):
        patient = self.get_patient_by_id(patient_id)
        if patient:
            patient.first_name = first_name
            patient.last_name = last_name
            patient.phone_number = phone_number
            self.session.commit()

    def delete_patient(self, patient_id):
        patient = self.get_patient_by_id(patient_id)
        if patient:
            self.session.delete(patient)
            self.session.commit()

    # Metoda find_all pentru a obține toți pacienții
    def find_all(self):
        return self.session.query(Patient).all()  # Returnează o listă cu toți pacienții