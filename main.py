# main.py
from dotenv import load_dotenv
import os
load_dotenv()
# 将 .env 文件加载逻辑集中在项目的主入口，其他模块直接使用 os.getenv 访问环境变量。

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from APP.routers import auth
from APP.routers import user
from APP.engine import uploadfiles,chatbot
import logging


app = FastAPI(debug=True)
UPLOAD_DIR = "uploads"
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# origins = [
#     "https://pieces-for-mom.netlify.app"
# ]


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Add more origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-ID"]
)

app.include_router(user.router)
app.include_router(auth.router, prefix="/auth")
app.include_router(uploadfiles.router)
app.include_router(chatbot.router)
# logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)