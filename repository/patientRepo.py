
from domain.patient import Patient
from sqlalchemy.orm import joinedload

from domain.teeth import Teeth


class PatientRepo:


    def __init__(self,session):
            self.session = session

    # CRUD pentru Patient
    def save(self,new_patient):

        self.session.add(new_patient)
        self.session.commit()
        return new_patient

    def get_patient_by_id(self, patient_id):
        print("repoPatient1")
        print(patient_id)
        return (
            self.session.query(Patient)
            .options(joinedload(Patient.teeths).joinedload(Teeth.issues))
            .filter(Patient.id == patient_id)
            .first()
        )

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

        return patient

    # Metoda find_all pentru a obține toți pacienții
    def find_all(self):
        return self.session.query(Patient).all()  # Returnează o listă cu toți pacienții