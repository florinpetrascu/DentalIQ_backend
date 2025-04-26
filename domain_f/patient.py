

class Patient:


    def __init__(self, id, firstName, lastName, phoneNumber):
        self.id = id
        self.firstName = firstName
        self.lastName = lastName
        self.phoneNumber = phoneNumber
        self.image = None
        self.teeth = []

    def set_image(self,image):
        self.image = image

    def set_teeth(self, teeth):
        self.teeth = teeth


    def set_firstName(self, firstName):
        self.firstName = firstName

    def set_lastName(self, lastName):
        self.lastName = lastName

    def set_phoneNumber(self, phoneNumber):
        self.phoneNumber = phoneNumber


