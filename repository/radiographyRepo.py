
from domain.radiography import Radiography

class RadiographyRepo:


    def __init__(self,session):
            self.session = session


    def find_by_patient_id(self, patient_id):
        return self.session.query(Radiography).filter(Radiography.patient_id == patient_id).first()

    def delete_by_patient_id(self,patient_id):
        count = self.session.query(Radiography).filter(Radiography.patient_id == patient_id).delete()
        self.session.commit()
        print("patient_id ",patient_id)
        print("radiografii sterse ",count)

    def save(self, new_image):
        self.session.add(new_image)
        self.session.commit()

        return new_image

