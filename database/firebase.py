# import os
#
# import firebase_admin
# from firebase_admin import credentials, firestore

# cred = credentials.Certificate("./dentaliq-api-firebase-adminsdk-fbsvc-b4bc3a4254.json")
# firebase_admin.initialize_app(cred)
#
# db = firestore.client()
#
#
# patients = db.collection('patients')
#
# docs = patients.stream()
#
# # Afișează toți pacienții
# for doc in docs:
#     print(f'{doc.id} => {doc.to_dict()}')
#
#
# print("Conectat la Firebase Firestore cu succes!")

# def get_db():
#     cred = credentials.Certificate("database/dentaliq-api-firebase-adminsdk-fbsvc-b4bc3a4254.json")
#     firebase_admin.initialize_app(cred)
#
#     db = firestore.client()
#
#     return db
#
# def get_db():
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     cred_path = os.path.join(base_dir, 'dentaliq-api-firebase-adminsdk-fbsvc-b4bc3a4254.json')
#
#     cred = credentials.Certificate(cred_path)
#
#     # Verifică dacă aplicația nu e deja inițializată
#     if not len(firebase_admin._apps):
#         firebase_admin.initialize_app(cred, {
#             'storageBucket': 'dentaliq-api.firebasestorage.app'
#         })
#
#     return firestore.client()
# # db = get_db()
# #
# # print(db)