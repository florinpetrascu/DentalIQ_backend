
from domain.radiography import Radiography
from repository.issueRepo import IssuesRepo
from repository.patientRepo import PatientRepo
from repository.radiographyRepo import RadiographyRepo
from repository.toothRepo import TeethRepo
import io

from database.firebase import bucket
from datetime import datetime,timedelta
class AiTasksService:

    def __init__(self , aiService, patientRepo:PatientRepo, toothRepo:TeethRepo, issueRepo:IssuesRepo, radiographyRepo:RadiographyRepo):

        self.aiService = aiService
        self.patientRepo = patientRepo
        self.toothRepo = toothRepo
        self.issueRepo = issueRepo
        self.radiographyRepo = radiographyRepo

    def save_image_to_firebase(self, image, patient_lastNname, patient_firstName):

        # Salvează temporar imaginea în memorie
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Creează un nume unic pentru imagine
        image_name = f"xrays/{patient_firstName}_{patient_lastNname}_radiography.png"

        # Trimite imaginea în Firebase Storage
        blob = bucket.blob(image_name)
        blob.upload_from_file(buffer, content_type='image/png')
        blob.make_public()
        # Generează un URL semnat valabil 7 zile
        #url = blob.generate_signed_url(expiration=timedelta(days=7))

        # Adaugă un query param ca să eviți caching-ul
        timestamp = datetime.utcnow().timestamp()
        return f"{blob.public_url}?t={int(timestamp)}"

    def analyse_panoramic_xray(self, image, patient_id):

        #aducem pacientul din baza de date
        patient = self.patientRepo.get_patient_by_id(patient_id)

        #salvam imaginea
        url = self.save_image_to_firebase(image, patient.last_name, patient.first_name)

        #aplicam AI pe imagine si obtinem lista de dinti
        teeth_list = self.aiService.get_teeths(image)

        print("from service ", teeth_list)

        #stergem dintii acestui pacient
        self.toothRepo.delete_by_patient_id(patient_id)

        #setam lista de dinti a pacientului
        for tooth in teeth_list:
            print(tooth)
            print("boli")
            print(tooth.issues)
            patient.addTooth(tooth)

        #salvam pacientul si il returam
        patient = self.patientRepo.save(patient) #Aici mi se adauga dintii dar nu mi se adauga

        #salvam url-ul radiografiei
        xray = Radiography(url=url,patient_id=patient.id)


        self.radiographyRepo.delete_by_patient_id(patient_id)

        self.radiographyRepo.save(xray)
        print("aici11")
        return patient



















