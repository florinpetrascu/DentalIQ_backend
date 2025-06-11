from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ai_service.aiService import AiService
from ai_service.aiServiceOnnx import AiServiceOnnx
from repository.issueRepo import IssuesRepo
from repository.radiographyRepo import RadiographyRepo
from repository.toothRepo import TeethRepo
from service.aiTasksService import AiTasksService

from repository.patientRepo import PatientRepo
from database.base import get_db
from sqlalchemy.orm import Session
import io
import base64
from PIL import Image
from dto_utils.dto_utils import get_patient_dto
from typing import Annotated
from auth.auth_provider import get_current_active_user, User

router = APIRouter(prefix="/api/ai", tags=["Ai"])
def get_ai_tasks_service(db: Session = Depends(get_db)) -> AiTasksService:
    patient_repo = PatientRepo(db)
    tooth_repo = TeethRepo(db)
    issue_repo = IssuesRepo(db)
    radiography_repo = RadiographyRepo(db)
    #ai_service = AiServiceOnnx("teeth_model.onnx","issues_model.onnx")
    ai_service = AiService("teeth_model.pt", "issues_model.pt")

    return AiTasksService(ai_service,patient_repo,tooth_repo,issue_repo,radiography_repo)


@router.post("/analyse")
async def analyse_image(
        current_user: Annotated[User, Depends(get_current_active_user)],
        patient: dict, service: AiTasksService = Depends(get_ai_tasks_service)):
    try:
        # Extrage datele pacientului și imaginea din JSON
        patient_data = patient.get("patient_data")
        #print('patient_data ', patient_data)
        image_base64 = patient.get("image")

        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided in the request")

        # Decodează imaginea din Base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        print("aici1")
        # Apelează serviciul pentru procesarea imaginii (detecția dinților)
        patient = service.analyse_panoramic_xray(image,patient_data['id'])
        #print("Patient inainte de dto")
        #print(patient.teeths)
        #print("aici2")
        patientDTO = get_patient_dto(patient)

        print(patientDTO)

        return JSONResponse(content=patientDTO)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

