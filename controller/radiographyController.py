from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from service.radiographyService import RadiographyService
from repository.radiographyRepo import RadiographyRepo
from database.base import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/radiographies", tags=["Radiographies"])
def get_radiography_service(db: Session = Depends(get_db)) -> RadiographyService:
    repo = RadiographyRepo(db)
    return RadiographyService(repo)

@router.get("/{patient_id}")
def get_radiography_by_patient_id(patient_id: int, service: RadiographyService = Depends(get_radiography_service)):
    # Căutăm radiografia pe baza ID-ului pacientului
    radiography = service.get_radiography_by_patient_id(patient_id)

    if radiography:
        return JSONResponse(
            content={"id": radiography.id, "url": radiography.url, "patient_id": radiography.patient_id})
    else:
        return JSONResponse(content={"detail": "Radiography not found"}, status_code=404)
