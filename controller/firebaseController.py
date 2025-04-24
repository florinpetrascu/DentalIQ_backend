

#from fastapi import FastAPI, HTTPException, UploadFile, File

# from fastapi.responses import JSONResponse
#
# from domain.patient import Patient
from repository.patientRepo import PatientRepo

#from repository.repository import Repository
from database.base import get_db
from fastapi import FastAPI, HTTPException

#from ai_service.aiService import AiService

from service.service import Service

# from dto_utils.dto_utils import get_patient_dto, get_patient_dto_for_firebase
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from fastapi.responses import JSONResponse
# from PIL import Image
# import numpy as np
#
# import io
# import base64





# FastAPI app
app = FastAPI()

# Creează o sesiune globală
session = next(get_db())

repo = PatientRepo(session)

issue_model_path = "issues_model.pt"
teeth_model_path = "teeth_model.pt"

#serviceAI = AiService(teeth_model_path, issue_model_path)
service = Service(repo)

@app.get("/api/patients")
def get_all_patients():

    # Serializare manuală
    result = service.get_all_patients()

    #return JSONResponse(content=result)
    return result

# @app.post("/api/patients")
# async def upload_patient(patientDTO: dict):
#
#
#     patient = Patient("nan", patientDTO['firstName'], patientDTO['lastName'], patientDTO['phoneNumber'])
#
#     patient = service.add_patient(patient)
#
#     newPatientDTO = get_patient_dto(patient)
#
#     return JSONResponse(content=newPatientDTO)
#
# @app.post("/api/upload")
# async def image_upload(patient: dict):
#     try:
#         # Extrage datele pacientului și imaginea din JSON
#         patient_data = patient.get("patient_data")
#         print('patient_data ',patient_data)
#         image_base64 = patient.get("image")
#
#
#         if not image_base64:
#             raise HTTPException(status_code=400, detail="No image provided in the request")
#
#         # Decodează imaginea din Base64
#         image_data = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_data)).convert("RGB")
#         original_image = np.array(image)
#
#
#         # Apelează serviciul pentru procesarea imaginii (detecția dinților)
#         teeth_list = service.get_tooth(image)
#
#
#
#         # Pregătește imaginea pentru răspuns
#         final_image = Image.fromarray(original_image.astype("uint8"))
#         img_io = io.BytesIO()
#         final_image.save(img_io, format="PNG")
#         img_io.seek(0)
#
#         # Encodează imaginea în Base64
#         encoded_image = base64.b64encode(img_io.getvalue()).decode("utf-8")
#
#         update_patient = Patient(patient_data['id'],patient_data['firstName'],patient_data['lastName'],patient_data['phoneNumber'])
#         #update_patient.set_image(encoded_image)
#         update_patient.set_teeth(teeth_list)
#
#         image_name = f"xrays/{patient_data['id']}_radiography.png"
#         patientDTO = service.update_patient(update_patient,img_io.getvalue(),image_name)
#
#
#         # Returnează imaginea procesată și datele despre dinți
#         #patientDTO = get_patient_dto_for_firebase(update_patient)
#         print(patientDTO)
#
#         return JSONResponse(content=patientDTO)
#
#     except Exception as e:
#         print(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#

if __name__ == "__main__":
    import uvicorn
    #print(service.get_all_patients())
    uvicorn.run("firebaseController:app", host="0.0.0.0", port=8000, reload=True)
