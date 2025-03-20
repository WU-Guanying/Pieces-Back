from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from ..database import get_db
from ..auth.auth_handler import get_current_active_user
from ..schemas import UserResponse, UserUpdate, UserR
from ..models import User
from sqlalchemy.orm import Session
from APP.utils.imageprocess import compress_image,MAX_IMAGE_SIZE
import os
import io
import shutil
import oss2
import mimetypes
import time

OSS_ACCESS_KEY_ID = os.getenv('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.getenv('OSS_ACCESS_KEY_SECRET')

auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)

endpoint = "https://oss-cn-beijing.aliyuncs.com"

bucketName = "pieces"
bucket = oss2.Bucket(auth, endpoint, bucketName)

router = APIRouter()
UPLOAD_DIR = "uploads/avatar"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
# Depends(get_current_active_user)： 
# 这是一个依赖项，FastAPI 会在执行 read_users_me 函数之前先执行 get_current_active_user，
# 并将其返回值作为 current_user 参数传递给 read_users_me。
@router.get("/users/me/", response_model=UserR)
async def read_users_me(current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    if not current_user.picture:  # 如果 picture 为 None 或空字符串
        current_user.picture = os.path.join(UPLOAD_DIR, "default_avatar.jpg")
        # db.commit()
        # db.refresh(current_user)
    return current_user

@router.post("/users/me/reset",response_model=UserR)
async def reset_users_me( user_update: str = Form(...),current_user: User = Depends(get_current_active_user), picture: UploadFile = File(None),db: Session = Depends(get_db)):
    import json
    try:
        user_update = json.loads(user_update)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in 'user_update'")
    user_update = UserUpdate(**user_update)
    tempourl = ''
    if user_update.nickname:
        current_user.nickname = user_update.nickname
    if user_update.email: 
        current_user.email = user_update.email
    if user_update.description:
        current_user.description = user_update.description
    if picture:
        # file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{picture.filename}")
        oss_object_name = f"{UPLOAD_DIR}/{current_user.id}_{picture.filename}"
        
        picture.file.seek(0)  # 复位文件指针
        # picture.file.read() 直接读取 UploadFile 对象的内容，返回的是 bytes。
        file_content = picture.file.read() # 读取原始图片内容（bytes）

        if len(file_content) > MAX_IMAGE_SIZE:
            # io.BytesIO(file_content) 创建一个类文件对象，让 PIL.Image.open() 可以处理它
            compressed_file = compress_image(io.BytesIO(file_content))
            compressed_file.seek(0)  
            file_content = compressed_file.read()# 重新读取压缩后的文件内容（bytes）
        mime_type, _ = mimetypes.guess_type(picture.filename)
        if mime_type is None:
            mime_type = "application/octet-stream"  # 兜底处理
        headers = {'Content-Type': mime_type}
        print('HEADER',headers)
        try:
            bucket.put_object(oss_object_name, file_content, headers=headers)
            current_user.picture = oss_object_name  # 只有上传成功才更新数据库字段
            tempourl = bucket.sign_url('GET', oss_object_name, 3600,params={'response-content-disposition': 'inline'}) + f"&t={int(time.time())}"
            current_user.tempourl = tempourl
            db.commit()  # 确保 OSS 上传成功后再提交数据库
            db.refresh(current_user)  # 确保数据是最新的
        except Exception as e:
            db.rollback()  # 避免部分提交
            raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
        

        # print("File Path:",file_path)
        # with open(file_path, "wb") as buffer:
        #     shutil.copyfileobj(picture.file, buffer)
        # current_user.picture = file_path
    
    print("user_update:", user_update)
    print("picture:", oss_object_name)
    
    return UserR(
        id=current_user.id,
        username=current_user.username,
        nickname=current_user.nickname,
        email=current_user.email,
        description=current_user.description,
        picture=current_user.picture,
        tempourl=tempourl  # 确保包含 tempourl
    )

@router.get("/users/me/refresh-url-header")
def refresh_temp_url(path: str = Query(..., description="The path of the image to refresh"), 
                     current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    if not path:
        raise HTTPException(status_code=400, detail="Invalid path")
    temp_url = None
    try:
        if "default_avatar.jpg" not in current_user.picture:
            print("refresh tempourl")
            temp_url = bucket.sign_url('GET', path.lstrip("/"), 3600,params={'response-content-disposition': 'inline'}) + f"&t={int(time.time())}" 
            current_user.tempourl = temp_url
            db.commit()
            db.refresh(current_user)
        return {"tempourl": temp_url}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to generate URL: {str(e)}")

# 头像图片将被保存在本地的 uploads/ 文件夹中，文件名格式是 "{user_id}_{original_filename}"，这样确保每个用户的头像文件是唯一的

# def get_current_active_user(current_user: User = Depends(get_current_user)):
#     if not current_user.is_active:
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user

# get_current_user：

# 提取 Token： 从请求头中提取 Token。
# 解码 Token： 使用 jwt.decode 解码 Token，验证其有效性。
# 获取用户信息： 从数据库中获取与 Token 中 sub（通常是用户名）对应的用户。
# 异常处理： 如果 Token 无效或用户不存在，抛出 HTTPException。
# get_current_active_user： 进一步检查用户是否处于激活状态，确保用户可以正常访问。

