from datetime import timedelta
import jwt
from jose.exceptions import JWTError
from fastapi import Depends, HTTPException, status, APIRouter
import os
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends, HTTPException, status, APIRouter
from sqlalchemy.orm import Session

from APP.auth.auth_handler import authenticate_user, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, get_user, get_password_hash, create_reset_token,pwd_context,SECRET_KEY,ALGORITHM
from APP.utils.sendemail import send_email
from ..database import get_db
from ..schemas import Token, UserCreate, UserResponse, PasswordResetRequest,PasswordReset
from ..models import User

router = APIRouter()

EMAIL_KEY = os.getenv("EMAIL_KEY")
@router.post("/register", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered.")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# login
@router.post("/token")
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
) -> Token:
    # 通过 OAuth2PasswordRequestForm 自动解析前端提交的表单数据。
    # form_data 会包含 username 和 password 属性。
    user = authenticate_user(db, form_data.username, form_data.password)
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

@router.post("/reset-request")
def request_password_reset(data: PasswordResetRequest, db: Session = Depends(get_db)):
    query = db.query(User).filter(User.username == data.username)
    if not query.first():
        raise HTTPException(status_code=404, detail="User not found")
    # user = db.query(User).filter(User.username == data.username,User.email == data.email).first()
    user = query.filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")
    
    reset_token = create_reset_token(username=user.username)
    reset_url = f"http://localhost:5173/reset-confirm?token={reset_token}"
    
    # Send the reset link to the user's email
    send_email(
    sender_email="guanying.x.wu@gmail.com",
    receiver_email=user.email,
    subject="Password Reset Request",
    body=f"Click the following link to reset your password: {reset_url}",
    smtp_server="smtp.gmail.com",
    port=587,
    password=EMAIL_KEY
    )
    
    return {"message": "Password reset email sent"}

@router.post("/reset-confirm")
def reset_password(data: PasswordReset, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(data.token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=400, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")
    
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update the user's password
    hashed_password = pwd_context.hash(data.new_password)
    user.password = hashed_password
    db.commit()
    return {"message": "Password reset successfully"}


