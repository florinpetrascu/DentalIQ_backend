


class Service:

    def __init__(self , serviceAI):
        self.serviceAI = serviceAI

    def get_tooth(self, image):
        return self.serviceAI.get_teeths(image)

    #def update_teeths(self , patient_id ,):