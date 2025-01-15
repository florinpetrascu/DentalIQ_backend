from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Declararea bazei de date
Base = declarative_base()

# Configurarea conexiunii la baza de date
DATABASE_URL = "sqlite:///dentalIQ.sqlite"
#jdbc:sqlite:dentalIQ.sqlite
# Crearea engine-ului pentru baza de date
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Configurarea sesiunii
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

print("Baza creeata")
# Funcție pentru obținerea unei sesiuni de bază de date
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
