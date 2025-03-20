from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from typing import List
from pydub import AudioSegment  # 处理音频
from PIL import Image  # 处理图片
import pandas as pd  # 处理 Excel 和 CSV
from fastapi import Depends, HTTPException, status, APIRouter
from ..auth.auth_handler import get_current_active_user
from ..models import User, FileUpload, Conversation, ConversationTurn
from ..schemas import BatchUploadConversationUpdate
from sqlalchemy.orm import Session
from APP.utils.imageprocess import compress_image,MAX_IMAGE_SIZE
import os
import io
from datetime import datetime
from ..database import get_db
from fastapi.responses import FileResponse
from sqlalchemy.sql.expression import func
import time
import mimetypes
import oss2
router = APIRouter()

OSS_ACCESS_KEY_ID = os.getenv('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.getenv('OSS_ACCESS_KEY_SECRET')

auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)

endpoint = "https://oss-cn-beijing.aliyuncs.com"

bucketName = "pieces"
bucket = oss2.Bucket(auth, endpoint, bucketName)


# 允许的文件类型
ALLOWED_FILE_TYPES = {
    "text": ["txt", "csv", "pdf", "xlsx","docx","md"],
    "image": ["jpg", "jpeg", "png"],
    "audio": ["mp3", "wav"]
}

# 上传文件存储路径
UPLOAD_DIRS = {
    "text": Path("uploads/text"),
    "image": Path("uploads/images"),
    "audio": Path("uploads/audio")
}

# 确保目录存在
for path in UPLOAD_DIRS.values():
    path.mkdir(parents=True, exist_ok=True)

@router.post("/users/batchuploads")
async def upload_files(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    uploaded_files = []
    upload_timestamps = {}

    for file in files:
        file_ext = file.filename.split(".")[-1].lower()

        # 确保文件类型合法
        # file_type = None
        # for key, extensions in ALLOWED_FILE_TYPES.items():
        #     if file_ext in extensions:
        #         file_type = key
        #         break
        file_type = next((key for key, extensions in ALLOWED_FILE_TYPES.items() if file_ext in extensions), None)

        if not file_type:
            raise HTTPException(status_code=400, detail=f"File Type Unsupported: {file.filename}")
        # / 运算符 在 pathlib.Path 对象中用于拼接路径（类似 os.path.join()）。
        # file.filename 是上传文件的名称（如 "example.pdf"）。
        # target_dir / file.filename 生成完整的文件路径。

        target_dir = UPLOAD_DIRS[file_type]
        # file_path = target_dir / file.filename # 本地
        oss_object_name = f"{target_dir}/{file.filename}"  # OSS 对象存储路径

        if file_type == "image":
                # 作用：将文件指针移动到文件的 末尾，然后返回当前位置（即文件的 总大小，以字节为单位）。
                # 为什么需要它？ 因为 file.file 是一个 文件对象（通常是 io.BytesIO 或 TemporaryFile），我们需要知道它的大小，决定是否要进行压缩。
                file_size = file.file.seek(0, io.SEEK_END)  # 获取文件大小
                file.file.seek(0)  # 复位文件指针

                if file_size > MAX_IMAGE_SIZE:
                    compressed_file = compress_image(file.file)
                    file_content = compressed_file.read()
                else:
                    file_content = file.file.read()
        else:
            file_content = file.file.read()        

        try:
            if file_type == 'image':
                mime_type, _ = mimetypes.guess_type(file.filename)
                if mime_type is None:
                    mime_type = "application/octet-stream"  # 兜底处理
                headers = {'Content-Type': mime_type}
                print('HEADER',headers)
                bucket.put_object(oss_object_name, file_content, headers=headers)
            else:
                bucket.put_object(oss_object_name, file_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OSS Upload Failed: {str(e)}")


        # 存储文件
        # with file_path.open("wb") as buffer: # 本地
        #     shutil.copyfileobj(file.file, buffer) # 本地

        upload_time = datetime.utcnow()
        # 存入数据库
        new_file = FileUpload(
            user_id=current_user.id,
            filename=file.filename,
            file_type=file_type,
            # file_path=str(file_path),
            file_path = oss_object_name,
            upload_time=upload_time,
        )
        db.add(new_file)
        uploaded_files.append(file.filename)
        upload_timestamps[file.filename] = upload_time
        
    # 最好在文件上传过程中捕获异常并在必要时进行回滚。
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="File upload failed, please try again later.")

    # files_new 是从数据库中查询出的 FileUpload 对象，
    # file 是 SQLAlchemy 对象而不是字典
    latest_files = []
    for filename in uploaded_files:
        latest_upload_time = db.query(func.max(FileUpload.upload_time)).filter(
            FileUpload.user_id == current_user.id,
            FileUpload.filename == filename
        ).scalar()

        if latest_upload_time:
            latest_file = db.query(FileUpload).filter(
                FileUpload.user_id == current_user.id,
                FileUpload.filename == filename,
                FileUpload.upload_time == latest_upload_time
            ).first()
            if latest_file:
                latest_files.append(latest_file)

    print("Latest uploaded files:", latest_files)

    return [
        {
            "id": file.id,
            "filename": file.filename,
            "file_type": file.file_type,
            "path": file.file_path,
            "upload_time": file.upload_time,
            "tempourl":bucket.sign_url('GET', file.file_path, 3600,params={'response-content-disposition': 'inline'})  if file.file_type == 'image' else ''
        }
        for file in latest_files
    ]

@router.post("/users/batchuploads-updateconversation")
async def updateconversation(req:BatchUploadConversationUpdate,current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    if req.conversation_id is None:
        new_conversation = Conversation(user_id=current_user.id, thread_id="thread_"+str(current_user.id)+"_"+str(int(time.time())))
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        conversation_id = new_conversation.id
    else:
        conversation_id = req.conversation_id
    print("batchuploads-updateconversation后端 conversation_id",conversation_id,req.dict_to_be_add)
    for uploadfile in req.dict_to_be_add:
        turn_number = db.query(ConversationTurn).filter_by(conversation_id=conversation_id).count() + 1
        user_turn = ConversationTurn(
            conversation_id=conversation_id,
            turn_number=turn_number,
            role="user",
            message=uploadfile['text'],
            state_snapshot={"text":uploadfile['text'],"type":uploadfile['type'],"path":uploadfile['path']}
        )
        db.add(user_turn)
    db.commit()
    return {"conversation_id":conversation_id}
   


@router.post("/users/uploads")
async def reset_users_me( file: UploadFile = File(...),current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    file_ext = file.filename.split(".")[-1].lower()

    # 确保文件类型合法
    file_type = None
    for key, extensions in ALLOWED_FILE_TYPES.items():
        if file_ext in extensions:
            file_type = key
            break

    if not file_type:
        raise HTTPException(status_code=400, detail="File Type Unsupported")
    
    target_dir = UPLOAD_DIRS[file_type]
    file_path = target_dir / file.filename #本地
    oss_object_name = f"{target_dir}/{file.filename}"
    
    # with file_path.open("wb") as buffer:#本地
    #     shutil.copyfileobj(file.file, buffer)#本地

    # 存入 OSS
    if file_type == "image":
            # 作用：将文件指针移动到文件的 末尾，然后返回当前位置（即文件的 总大小，以字节为单位）。
            # 为什么需要它？ 因为 file.file 是一个 文件对象（通常是 io.BytesIO 或 TemporaryFile），我们需要知道它的大小，决定是否要进行压缩。
            file_size = file.file.seek(0, io.SEEK_END)  # 获取文件大小
            file.file.seek(0)  # 复位文件指针

            if file_size > MAX_IMAGE_SIZE:
                compressed_file = compress_image(file.file)
                file_content = compressed_file.read()
            else:
                file_content = file.file.read()
    else:
        file_content = file.file.read()

    try:
        if file_type == 'image':
            mime_type, _ = mimetypes.guess_type(file.filename)
            if mime_type is None:
                mime_type = "application/octet-stream"  # 兜底处理
            headers = {'Content-Type': mime_type}
            print('HEADER',headers)
            bucket.put_object(oss_object_name, file_content, headers=headers)
        else:
            bucket.put_object(oss_object_name, file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OSS Upload Failed: {str(e)}")

    # 存入数据库
    new_file = FileUpload(
        user_id=current_user.id,
        filename=file.filename,
        file_type=file_type,
        # file_path=str(file_path),
        file_path=oss_object_name,
        upload_time=datetime.utcnow(),
    )
    db.add(new_file)
    db.commit()

    return {"message": "Upload Succeeded!", "filename": file.filename, "path": str(file_path)}

@router.get("/users/files")
def get_user_files(current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    files = db.query(FileUpload).filter(FileUpload.user_id == current_user.id).all()
    if not files:
        return {"message": "No File Found."}
    
    return [
        {"id": file.id, "filename": file.filename, "file_type": file.file_type, "path": file.file_path, "upload_time": file.upload_time}
        for file in files
    ]


@router.get("/users/files-download/{file_id}")
def download_file(file_id: int, current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    file = db.query(FileUpload).filter(FileUpload.id == file_id,FileUpload.user_id == current_user.id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File Not Found.")
    
    ###
    oss_object_name = file.file_path
    local_file_path = Path("/Users/chirp/Downloads") / oss_object_name
    local_file_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        bucket.get_object_to_file(oss_object_name,str(local_file_path))
    except oss2.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="File Lost in OSS.")
    ###
    
    # file_path = Path(file.file_path)
    # print("Back file path:",file_path)
    # if not file_path.exists():
    #     raise HTTPException(status_code=404, detail="File Lost.")

    # 返回文件，用于 下载 或 提供静态资源。
    # 自动设置 HTTP 头，确保正确的 Content-Disposition 和 Content-Type。 
    return FileResponse(local_file_path, filename=file.filename)

@router.delete("/users/files-delete/{file_id}")
def delete_file(file_id: int,current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    file = db.query(FileUpload).filter(FileUpload.id == file_id,FileUpload.user_id == current_user.id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File Not Found.")
    
    # 删除物理文件
    # file_path = Path(file.file_path) #本地
    # if file_path.exists(): #本地
    #     os.remove(file_path) #本地
    
    # 在oss中删除
    try:
    # 尝试删除对象
        bucket.delete_object(file.file_path)
        print(f"Object {file.file_path} deleted successfully.")
    except oss2.exceptions.OssError as e:
        # 捕获 OSS 相关的错误，并输出详细的错误信息
        print(f"Failed to delete object {file.file_path}. OSS Error: {e}")
    except Exception as e:
        # 捕获其他所有类型的异常
        print(f"An error occurred while deleting object {file.file_path}: {e}")

    
    # 删除数据库记录
    db.delete(file)
    db.commit()

    return {"message": "File Deleted.", "file_id": file_id}

# class FileUploadRequest(BaseModel):
#     dict_to_be_add: List[dict]  # 复杂 JSON 数据

# @app.post("/users/batchuploads-updateconversation")
# async def update_conversation(
#     conversation_id: int = Query(..., description="Conversation ID"), 
#     request_body: FileUploadRequest = Depends()
# ):
#     return {
#         "message": "Received",
#         "conversation_id": conversation_id,
#         "dict_to_be_add": request_body.dict_to_be_add
#     }



# const handleFileUpload = async (files) => {
#     if (files.length === 0) {
#         return;
#     }

#     const formData = new FormData();
#     Array.from(files).forEach(file => formData.append("files", file));

#     let newMessages;
#     try {
#         setBatchUploadStatus("Uploading...");
#         const response = await axios.post(
#             `${API_URL}/users/batchuploads-updateconversation`, 
#             {
#                 dict_to_be_add: newMessages // 复杂 JSON 数据
#             },
#             {
#                 headers: {
#                     'Content-Type': 'application/json',
#                     Authorization: `Bearer ${token}`,
#                 },
#                 params: { 
#                     conversation_id: selectedTitle ? selectedTitle.conversation_id : conversationId
#                 }
#             }
#         );

#         console.log("Response data:", response.data);
#         setBatchUploadStatus("Upload Succeed");

#     } catch (error) {
#         setBatchUploadStatus("Upload Failed: " + (error?.response?.data?.detail || error.message));
#     }
# };


