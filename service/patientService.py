from domain.patient import Patient


class PatientService:

    def __init__(self,patientRepo):
        self.patientRepo = patientRepo

    def get_all_patients(self):
        return self.patientRepo.find_all()

    def add_patient(self , first_name, last_name, phone_number):
        patient = Patient(first_name = first_name,last_name = last_name, phone_number = phone_number)
        return self.patientRepo.save(patient)

    def get_patient_by_id(self,patient_id):
        return self.patientRepo.get_patient_by_id(patient_id)