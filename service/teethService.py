
from repository.toothRepo import TeethRepo


class TeethService:

    def __init__(self,toothRepo:TeethRepo):

        self.toothRepo = toothRepo


    def get_teeth_by_patient_id(self,patient_id):

        return self.toothRepo.find_by_patient_id(patient_id)


