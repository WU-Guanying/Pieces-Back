# models.py
from sqlalchemy import Column, Integer, String, Boolean,Text, Enum, TIMESTAMP, ForeignKey, JSON,DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True)
    email = Column(String(255), index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    description = Column(String(1000), default="")  # 用户简介，默认空字符串
    picture = Column(String(255), default="")  # 头像图片 URL，默认空字符串
    nickname = Column(String(255),default="momo")
    tempourl = Column(Text, nullable=True)

class FileUpload(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String(255))
    file_type = Column(String(50))
    file_path = Column(String(255))
    upload_time = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    thread_id = Column(String(100), nullable=False)  # 对应 LangGraph 的 thread_id
    started_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    title = Column(Text, nullable=False, default=lambda: f"Topic Created At {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

class ConversationTurn(Base):
    __tablename__ = 'conversation_turns'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    turn_number = Column(Integer, nullable=False)
    role = Column(Enum('user', 'ai'), nullable=False)
    message = Column(Text, nullable=True)
    state_snapshot = Column(JSON, nullable=True)  # 存储 LangGraph 的状态
    created_at = Column(TIMESTAMP, server_default=func.now())
