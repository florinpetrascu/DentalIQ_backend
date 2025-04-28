
from fastapi import FastAPI
from models_provider.download_models import download_file
import os
from fastapi.middleware.cors import CORSMiddleware


# de modificat aici pentru deploy!!!!!
from radiographyController import router as radiography_router
from patientController import  router as patient_router
from aiTasksController import router as ai_router
# FastAPI app
app = FastAPI()


origins = [
    "http://localhost:3000",  # local
    "https://dental-iq-frontend.vercel.app"  # Vercel frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # sau ["*"] pentru orice (nu recomandat în producție)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(radiography_router)
app.include_router(patient_router)
app.include_router(ai_router)


def load_models():

        issue_model_url = "https://raw.githubusercontent.com/florinpetrascu/pdm_backend/main/issues_model.pt"
        teeth_model_url = "https://raw.githubusercontent.com/florinpetrascu/pdm_backend/main/teeth_model.pt"

        issue_model_path = "issues_model.pt"
        teeth_model_path = "teeth_model.pt"

        if not os.path.exists("issues_model.pt"):
            download_file(issue_model_url, "issues_model.pt")

            print("Fisierul issues_model.pt incarcat cu success")
        else:
            print("Fișierul issues_model.pt există deja. Nu se descarcă din nou.")

        if not os.path.exists("teeth_model.pt"):
            download_file(teeth_model_url, "teeth_model.pt")

            print("Fisierul teeth_model.pt incarcat cu success")
        else:
            print("Fișierul issues_model.pt există deja. Nu se descarcă din nou.")


if __name__ == "__main__":
    import uvicorn
    #print(service.get_all_patients())
    load_models()
    uvicorn.run("firebaseController:app", host="0.0.0.0", port=8000, reload=True)

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
