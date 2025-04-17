


class Service:

    def __init__(self , serviceAI, patientRepo):
        self.serviceAI = serviceAI
        self.patientRepo = patientRepo

    def get_tooth(self, image):
        return self.serviceAI.get_teeths(image)

    def add_patient(self , patient):
        return self.patientRepo.add_patient(patient)

    def get_all_patients(self):
        return self.patientRepo.get_all_patientsDTO()

    def update_patient(self, updated_data, image_bytes, firebase_path: str):
        return self.patientRepo.update_patient(updated_data, image_bytes, firebase_path)
