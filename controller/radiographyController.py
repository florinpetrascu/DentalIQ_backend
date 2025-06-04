from fastapi import APIRouter, Depends,  HTTPException, Request
from fastapi.responses import JSONResponse, Response
from service.radiographyService import RadiographyService
from repository.radiographyRepo import RadiographyRepo
from database.base import get_db
from sqlalchemy.orm import Session
import requests

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

@router.post("/proxy-image")
async def proxy_image(request: Request):
    body = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        response = requests.get(url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "image/png")
        return Response(content=response.content, media_type=content_type)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")