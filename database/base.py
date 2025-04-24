from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Declararea bazei de date
Base = declarative_base()

# Configurarea conexiunii la baza de date
DATABASE_URL = "postgresql://cek6h0n4ffe7ur.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/d5opo0smtkt17n"
#jdbc:sqlite:dentalIQ.sqlite
# Crearea engine-ului pentru baza de date

engine = create_engine('postgresql+psycopg2://uf3f8puk155hqr:pb6ce338c3376a3ce04fccc639d700efd92fd63307a3359c5c21f2c068c77ed09@cek6h0n4ffe7ur.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com/d5opo0smtkt17n')
#engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

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
