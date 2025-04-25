from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from dto_utils.dto_utils import get_simple_patient_dto
from service.patientService import PatientService
from repository.patientRepo import PatientRepo
from database.base import get_db
from sqlalchemy.orm import Session

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


    #patient = Patient("nan", patientDTO['firstName'], patientDTO['lastName'], patientDTO['phoneNumber'])

    patient = service.add_patient(patientDTO['firstName'], patientDTO['lastName'], patientDTO['phoneNumber'])

    newPatientDTO = get_simple_patient_dto(patient)

    return JSONResponse(content=newPatientDTO)

    #return patient