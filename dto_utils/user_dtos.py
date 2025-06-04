

from domain.user import User

def get_user_dto(user: User):

        return {
            "id": user.id,
            "nume": user.nume,
            "prenume": user.prenume,
            "username": user.username,
            "full_name": user.prenume + ' ' + user.nume,
            "hashed_password": user.hashed_password,
            "email": user.email,
            "disabled": not user.activ,  # echivalent cu disabled din UserInDB
        }