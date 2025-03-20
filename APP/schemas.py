# schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from pydantic import Field
import os


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: str | None = None


class UserUpdate(BaseModel):
    nickname: Optional[str] = Field(None, max_length=50, description="Maximum length is 50 characters.")
    email: Optional[EmailStr] = None
    description: Optional[str] = Field(None, max_length=200, description="Maximum length is 200 characters.")

class UserR(BaseModel):
    id: int
    username:str
    nickname: str
    email: EmailStr
    description: Optional[str] = None
    picture: Optional[str] = None
    tempourl: Optional[str] = None
    
    class Config:
        from_attributes = True

class UserInDB(UserResponse):
    hashed_password: str

class PasswordResetRequest(BaseModel):
    email: str
    username: str

class PasswordReset(BaseModel):
    token: str
    new_password: str

class ChatRequest(BaseModel):
    conversation_id: Optional[int] = None  # 如果为空则新建会话
    message: str
    files: List[dict]

class ChatRequestImage(BaseModel):
    conversation_id: Optional[int] = None  # 如果为空则新建会话
    message: str

class ChatResponse(BaseModel):
    reply: str
    conversation_id: int

class ChatResponseImage(BaseModel):
    reply: str
    conversation_id: int
    tempourl:Optional[str] = str
    type:Optional[str] = str
    path:Optional[str] = str


class TitleUpdateRequest(BaseModel):
    title: str

class BatchUploadConversationUpdate(BaseModel):
    conversation_id: Optional[int] = None 
    dict_to_be_add: List[dict]

class TextInput(BaseModel):
    text: str
    

class ChatRequestAudio(BaseModel):
    conversation_id: int | None = None
    message: str

class ChatResponseAudio(BaseModel):
    reply: str
    conversation_id: int
    tempourl: str | None = None
    type: str | None = None
    path: str | None = None

class DeleteAudioRequest(BaseModel):
    file_path: str

# 如果想让 uploadfile 变成 Pydantic 模型对象
# class UploadFileSchema(BaseModel):
#     text: str
#     type: str
#     path: str

# class BatchUploadConversationUpdate(BaseModel):
#     conversation_id: Optional[int] = None 
#     dict_to_be_add: List[UploadFileSchema]  
# 这样，在 for uploadfile in req.dict_to_be_add: 里面，uploadfile 就会是一个 UploadFileSchema 对象，
# 而不是 dict，访问 uploadfile.text、uploadfile.type 时就可以使用点运算符 (.) 而不是 [''] 了。