import os
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Calea către fișierul de autentificare
base_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(base_dir, 'dentaliq-api-firebase-adminsdk-fbsvc-b4bc3a4254.json')

# Inițializare o singură dată
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'dentaliq-api.firebasestorage.app'
    })

# Obiecte globale
db = firestore.client()
bucket = storage.bucket()
