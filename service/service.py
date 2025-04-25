from domain.patient import Patient


class Service:

    def __init__(self , patientRepo):
        #self.serviceAI = serviceAI
        self.patientRepo = patientRepo

    # def get_tooth(self, image):
    #     return self.serviceAI.get_teeths(image)

    def add_patient(self , first_name, last_name, phone_number):
        patient = Patient(first_name = first_name,last_name = last_name, phone_number = phone_number)
        return self.patientRepo.save(patient)

    def get_all_patients(self):
        return self.patientRepo.find_all()

    def update_patient(self, updated_data, image_bytes, firebase_path: str):
        return self.patientRepo.update_patient(updated_data, image_bytes, firebase_path)
