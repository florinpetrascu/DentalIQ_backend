
from fastapi import FastAPI
from models_provider.download_models import download_file
import os
from fastapi.middleware.cors import CORSMiddleware


# de modificat aici pentru deploy!!!!!
# from controller.radiographyController import router as radiography_router
# from controller.patientController import router as patient_router
# from controller.aiTasksController import router as ai_router

from controller.radiographyController import router as radiography_router
from controller.patientController import router as patient_router
from controller.aiTasksController import router as ai_router
from controller.loginController import router as login_router

# FastAPI app
app = FastAPI()


origins = [
    "http://localhost:3000",  # local
    "https://dental-iq-frontend.vercel.app"  # Vercel frontend

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sau ["*"] pentru orice (nu recomandat în producție)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(login_router)
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
