

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
import shutil


import os

from ai_service.aiService import AiService
from domain.note import Note
from domain.patient import Patient
from domain.teeth import Teeth
from domain.issue import Issue
from database.base import Base,engine,get_db
from service.service import Service
from dto_utils.dto_utils import get_patient_dto
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import io
import base64

Base.metadata.create_all(bind=engine)
# Creează tabelele în baza de date
print("Tabele detectate:", Base.metadata.tables.keys())
Base.metadata.create_all(bind=engine)
print("Tabelele au fost create cu succes!")
print("Calea absolută a bazei de date:", os.path.abspath("dentalIQ.sqlite"))
# FastAPI app
app = FastAPI()

teethModelPath = "../ai_service/teeth_model.pt"
issueModelPath = "../ai_service/issues_model.pt"

# Instantiate the AiService with the models
serviceAI = AiService(teethModelPath, issueModelPath)
service = Service(serviceAI)
@app.get("/api/patients")
def get_all_patients():
    db = next(get_db())
    patients = db.query(Patient).all()

    # Serializare manuală
    result = []
    for patient in patients:
        patient_dto = get_patient_dto(patient)
        result.append(patient_dto)

    return JSONResponse(content=result)

@app.post("/api/patients")
async def upload_patient(patientDTO: dict):
    db = next(get_db())

    patient = Patient()
    patient.firstName = patientDTO['firstName']
    patient.lastName = patientDTO['lastName']
    patient.phoneNumber = patientDTO['phoneNumber']

    db.add(patient)
    db.commit()

    print("patient_id" , patient.id)

    newPatientDTO = get_patient_dto(patient)

    return JSONResponse(content=newPatientDTO)

@app.post("/addNote")
def add_note(patient_id: int, tooth_id: int, text: str):
    db = next(get_db())
    tooth = db.query(Teeth).filter(Teeth.id == tooth_id).first()
    if not tooth:
        raise HTTPException(status_code=404, detail="Tooth not found")
    new_note = Note(text=text, tooth_id=tooth_id)
    db.add(new_note)
    db.commit()
    db.refresh(new_note)
    return new_note

@app.post("/loadImage")
def load_image(patient_id: int, file: UploadFile = File(...)):
    db = next(get_db())
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    file_path = f"images/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    patient.image = file_path
    db.commit()
    return {"filename": file.filename, "file_path": file_path}




@app.post("/api/upload")
async def image_upload(patient: dict):
    try:
        # Extrage datele pacientului și imaginea din JSON
        patient_data = patient.get("patient_data")
        print('patient_data ',patient_data)
        image_base64 = patient.get("image")

        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided in the request")

        # Decodează imaginea din Base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_image = np.array(image)


        # Apelează serviciul pentru procesarea imaginii (detecția dinților)
        teeth_list = service.get_tooth(image)



        # Pregătește imaginea pentru răspuns
        final_image = Image.fromarray(original_image.astype("uint8"))
        img_io = io.BytesIO()
        final_image.save(img_io, format="PNG")
        img_io.seek(0)

        # Encodează imaginea în Base64
        encoded_image = base64.b64encode(img_io.getvalue()).decode("utf-8")
        db = next(get_db())

        update_patient = db.query(Patient).filter(Patient.id == patient_data['id']).first()

        existingTeeth = db.query(Teeth).filter(Teeth.patient_id == update_patient.id)

        #stergem bolile  aferente pacientului
        for tooth in existingTeeth:
            db.query(Issue).filter(Issue.tooth_id == tooth.id).delete()

        #stergem dintii aferenti pacientului
        db.query(Teeth).filter(Teeth.patient_id == update_patient.id).delete()
        print('update_patient ' , update_patient.id , ' ', update_patient.firstName)

        if not update_patient:
            raise HTTPException(status_code=404, detail="Pacientul nu există.")

        update_patient.firstName = patient_data['firstName']
        update_patient.lastName = patient_data['lastName']
        update_patient.phoneNumber = patient_data['phoneNumber']
        update_patient.image = encoded_image



        for tooth in teeth_list:
            tooth.patient_id = update_patient.id
            tooth.polygon = str(tooth.polygon)
            db.add(tooth)  # Adaugă dinții noi în sesiunea bazei de date

        db.commit()



        # Returnează imaginea procesată și datele despre dinți
        patientDTO = get_patient_dto(update_patient)
        return JSONResponse(content=patientDTO)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("controller:app", host="127.0.0.1", port=8000, reload=True)
#
#
