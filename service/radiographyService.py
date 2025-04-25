


class RadiographyService:

    def __init__(self, radiographyRepo):
        self.radiographyRepo = radiographyRepo


    def get_radiography_by_patient_id(self, patient_id):
        return self.radiographyRepo.find_by_patient_id(patient_id)

