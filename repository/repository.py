# from datetime import datetime
#
# import database.firebase
# from dto_utils.dto_utils import get_patient_dto , get_patient_dto_for_firebase
# from firebase_admin import storage
# import json
# class Repository:
#
#     def __init__(self):
#         self.db = database.firebase.get_db()
#
#     def get_all_patientsDTO(self):
#         patients_ref = self.db.collection('patients')
#         docs = patients_ref.stream()
#
#         patients = []
#         for doc in docs:
#             data = doc.to_dict()
#             data['id'] = doc.id
#             # print("doc")
#             # print(doc.to_dict())
#             # data['teeths'][i] = json.loads(doc.to_dict()[''])
#             patients.append(data)
#
#         return patients
#
#     def add_patient(self, patient):
#
#         #patient e de tipul Patient
#         patientDTO = get_patient_dto(patient)
#
#         doc_ref = self.db.collection("patients").add(patientDTO)
#         print("Pacient adăugat cu ID:", doc_ref[1].id)
#
#         patient.id = doc_ref[1].id
#
#         return patient
#
#
#     def update_patient(self, updated_data, image_bytes, firebase_path: str):
#
#         print("upload image to firebase")
#         image_url = self.upload_image_to_firebase(image_bytes, firebase_path)
#         updated_data.set_image(image_url)
#
#         print("image uploaded to firebase")
#
#         patient_ref = self.db.collection('patients').document(updated_data.id)
#         updated_data_DTO = get_patient_dto_for_firebase(updated_data)
#
#         print("Pe aici")
#
#         print(patient_ref.get())
#         patient_ref.update(updated_data_DTO)
#
#         updated_patient_snapshot = patient_ref.get()
#         updated_patient = updated_patient_snapshot.to_dict()
#         updated_patient["id"] = updated_patient_snapshot.id  # Adaugă ID-ul dacă vrei
#
#         return updated_patient
#
#     def upload_image_to_firebase(self, image_bytes, firebase_path: str) -> str:
#         bucket = storage.bucket()
#         blob = bucket.blob(firebase_path)
#         blob.upload_from_string(image_bytes, content_type='image/png')
#         blob.make_public()  # faci imaginea accesibilă public
#
#         # Adaugă un query param ca să eviți caching-ul
#         timestamp = datetime.utcnow().timestamp()
#         return f"{blob.public_url}?t={int(timestamp)}"
#
#
#
