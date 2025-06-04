from fastapi import APIRouter, Depends, HTTPException
from service.patientService import PatientService
from repository.patientRepo import PatientRepo

from datetime import datetime, timedelta, timezone
from typing import Annotated


from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


from database.base import get_db
from sqlalchemy.orm import Session

from auth.auth_provider import Token, authenticate_user_db, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, User, get_current_active_user, get_user_service
from repository.userRepo import UserRepo
from service.userService import UserService




router = APIRouter(prefix="/api/login", tags=["Login"])




@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> Token:


    user = authenticate_user_db(user_service, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user



