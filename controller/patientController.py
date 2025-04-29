from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from dto_utils.dto_utils import get_simple_patient_dto, get_patient_dto
from service.patientService import PatientService
from repository.patientRepo import PatientRepo
from database.base import get_db
from sqlalchemy.orm import Session
from models_provider.download_models import download_file
import os
router = APIRouter(prefix="/api/patients", tags=["Patients"])
def get_patient_service(db: Session = Depends(get_db)) -> PatientService:
    repo = PatientRepo(db)
    return PatientService(repo)


@router.get("/")
def get_all_patients( service: PatientService = Depends(get_patient_service)):

    # Serializare manuală
    result = service.get_all_patients()

    #return JSONResponse(content=result)
    return result

@router.post("/")
async def upload_patient(patientDTO: dict,  service: PatientService = Depends(get_patient_service)):

    #descarcare onnx files

    load_models()


    #patient = Patient("nan", patientDTO['firstName'], patientDTO['lastName'], patientDTO['phoneNumber'])

    patient = service.add_patient(patientDTO['firstName'], patientDTO['lastName'], patientDTO['phoneNumber'])

    newPatientDTO = get_simple_patient_dto(patient)

    return JSONResponse(content=newPatientDTO)

    #return patient

@router.get("/{patient_id}")
def get_patient(patient_id: int, service: PatientService = Depends(get_patient_service)):
    patient = service.get_patient_by_id(patient_id)
    if not patient:
        print("Eroare aici")
        raise HTTPException(status_code=404, detail="Patient not found")

    patientDTO = get_patient_dto(patient)

    print(patientDTO)

    return JSONResponse(content=patientDTO)


def load_models():
    issue_model_url = "https://github.com/florinpetrascu/onnx_models/releases/download/v1.0/issues_model.onnx"
    teeth_model_url = "https://github.com/florinpetrascu/onnx_models/releases/download/v1.0/teeth_model.onnx"

    print("load models")

    if not os.path.exists("issues_model.onnx"):
        download_file(issue_model_url, "issues_model.onnx")

        print("Fisierul issues_model.onnx incarcat cu success")
    else:
        print("Fișierul issues_model.onnx există deja. Nu se descarcă din nou.")

    if not os.path.exists("teeth_model.onnx"):
        download_file(teeth_model_url, "teeth_model.onnx")

        print("Fisierul teeth_model.onnx incarcat cu success")
    else:
        print("Fișierul issues_model.onnx există deja. Nu se descarcă din nou.")
