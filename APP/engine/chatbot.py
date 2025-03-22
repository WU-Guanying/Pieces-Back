from ..database import get_db
from io import BytesIO
from tempfile import NamedTemporaryFile
from ..schemas import ChatRequest,ChatResponseImage,TitleUpdateRequest,ChatRequestImage,TextInput,DeleteAudioRequest
from ..models import User, Conversation, ConversationTurn
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from fastapi import Depends, HTTPException, status, APIRouter,UploadFile, File
from APP.utils.imageprocess import compress_image,MAX_IMAGE_SIZE
from langchain_core.prompts import PromptTemplate
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse,JSONResponse
from typing import AsyncGenerator
from ..auth.auth_handler import get_current_active_user
from APP.utils.imageprocess import encode_image
from pathlib import Path
import mimetypes
import requests
import re
import time
import os
import oss2
import io
from openai import OpenAI

router = APIRouter()

OPENAI_KEY = os.getenv("OPENAI_KEY")
LANGCHAIN_KEY = os.getenv("LANGCHAIN_KEY")
OSS_ACCESS_KEY_ID = os.getenv('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.getenv('OSS_ACCESS_KEY_SECRET')

auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)

endpoint = "https://oss-cn-beijing.aliyuncs.com"

# https://<你的存储桶名>.oss-cn-beijing.aliyuncs.com/...（Bucket 设为 "公共读"）

bucketName = "pieces"
bucket = oss2.Bucket(auth, endpoint, bucketName)

UPLOAD_DIR = "uploads/images-generation/"
UPLOAD_DIR_AUDIO = "uploads/audio-generation"

os.environ['OPENAI_API_KEY']=OPENAI_KEY
os.environ['LANGCHAIN_API_KEY']=LANGCHAIN_KEY
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
client = OpenAI()
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# pdf
# loader = PyPDFLoader(
#     "./example_data/layout-parser-paper.pdf",
#     mode="page",
#     images_inner_format="markdown-img",
#     images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o", max_tokens=1024)),
# )
# docs = loader.load()
# print(docs[5].page_content)

# csv
# loader = CSVLoader(
#     file_path="./example_data/mlb_teams_2012.csv",
#     csv_args={
#             "delimiter": ",",
#             "quotechar": '"',
#             "fieldnames": ["MLB Team", "Payroll in millions", "Wins"],
#         },
#     source_column="Team")
# data = loader.load()

# xlsx
# loader = UnstructuredExcelLoader("./example_data/stanley-cups.xlsx", mode="elements")
# docs = loader.load()

# word
# loader = Docx2txtLoader("./example_data/fake.docx")
# data = loader.load()
FILE_TYPE_LOADERS = {
    'csv': CSVLoader,
    'pdf': PyPDFLoader,
    'docx': Docx2txtLoader,
    'xlsx': UnstructuredExcelLoader,
    'txt':TextLoader,
    'md':TextLoader
}

# 获取文件扩展名
def get_file_extension(file_path: str) -> str:
    return file_path.split('.')[-1].lower()

def load_file_content(file_path: str, bucket):
    file_extension = get_file_extension(file_path)
    file_obj = bucket.get_object(file_path)
    file_content = file_obj.read()
    with NamedTemporaryFile(delete=False, mode='wb') as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    # 如果文件类型有对应的 loader，使用相应的加载器
    loader_class = FILE_TYPE_LOADERS.get(file_extension)
    if loader_class:
        loader = loader_class(temp_file_path)
        document = loader.load()
        os.remove(temp_file_path)
        return document
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

@router.get("/users/history")
def history( conversation_id: int,current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    turns = db.query(ConversationTurn).filter_by(conversation_id=conversation_id).order_by(ConversationTurn.created_at.asc()).all()
    if not turns:
        return []
    # state_snapshot 是 JSON 类型，存储在数据库中，相当于一个 dict。
    # 在数据库里，它是 JSON 类型（类似 JSON 字符串）。
    # SQLAlchemy 取出来后，它会被解析成 Python dict，所以 t.state_snapshot 在 Python 代码里是一个字典。
    # 在 Python 代码中，SQLAlchemy 会把 JSON 字段解析成 Python 字典 (dict)。
    # t.state_snapshot 是一个字典，而 get() 是字典的安全访问方
    return [{
            "role": t.role,
            "text": t.message, 
            "type": t.state_snapshot.get("type") if t.state_snapshot else None,
            "path": t.state_snapshot.get("path") if t.state_snapshot else None,
            "tempourl": bucket.sign_url('GET', t.state_snapshot.get("path"), 3600,params={'response-content-disposition': 'inline'}) if t.state_snapshot else None, 
        } for t in turns]
# FastAPI 自动将 Python dict 列表转换为 JSON，返回给前端。
# axios 自动解析 JSON，将它转换为 JavaScript 对象数组，存入 messages：

@router.get("/users/conversations")
def select_conversation(current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    turns = db.query(Conversation).filter_by(user_id=current_user.id).order_by(Conversation.updated_at.desc()).all()
    if not turns:  # 没有对话记录
        return []
    return [{"conversation_id":t.id,"title":t.title,"started_at": t.started_at,"updated_at":t.updated_at} for t in turns]

@router.delete("/users/conversations-delete/{conversation_id}")
def delete_conversation(conversation_id: int,current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    
    db.query(ConversationTurn).filter(ConversationTurn.conversation_id == conversation_id).delete()
    db.query(Conversation).filter(Conversation.id == conversation_id,Conversation.user_id == current_user.id).delete()
    
    db.commit()
    
    return {"message": "Conversastion Deleted.", "conversation_id": conversation_id}

@router.post("/users/title-edit/{conversationId}")
def edit_conversation_title(conversationId: int,request:TitleUpdateRequest,current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(Conversation.id == conversationId,Conversation.user_id == current_user.id).first()
    # print("chatbot conversation", conversation)
    # print("chatbot raw request:", request) 
    # 如果参数是 Pydantic 模型（BaseModel），FastAPI 会自动从 body 解析它。
    # 如果参数是字符串（title: str），FastAPI 只能从查询参数或路径参数获取它，不能从 body 里获取。
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation.title = request.title 
    db.add(conversation)  # 添加到session中（对于更新操作这一步通常是必须的）
    db.commit()  # 提交更新
    db.refresh(conversation)  # 刷新对象以确保包含最新的数据库值
    return {"message": "Conversastion Title Updated.", "conversation_id": conversationId, "title":request.title}

# @router.post("/users/chat", response_model=ChatResponse)
@router.post("/users/chat", response_class=StreamingResponse)
def chat_endpoint(req: ChatRequest, current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    # 查询用户是否存在（省略验证）

    # 如果 conversation_id 为空则新建会话
    if req.conversation_id is None:
        new_conversation = Conversation(user_id=current_user.id, thread_id="thread_"+str(current_user.id)+"_"+str(int(time.time())))
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        conversation_id = new_conversation.id
    else:
        conversation_id = req.conversation_id

    # 计算 turn_number（简单查询该会话下的轮次计数）
    turn_number = db.query(ConversationTurn).filter_by(conversation_id=conversation_id).count() + 1
    
    # conversation = db.query(Conversation).filter_by(id=conversation_id).first()
    turns = db.query(ConversationTurn).filter_by(conversation_id=conversation_id).order_by(ConversationTurn.created_at.asc()).all() 
    # full_message = [{"role": t.role, "text": t.message} for t in turns[-200:]]# + [{"role":"user","text":current_turn_message}]

    # dialog_history = ""
    # for message in full_message:
    #     role = message['role']
    #     text = message['text']
    #     dialog_history += f"({role.capitalize()}): {text}\n"
    dialog_history = "\n".join([f"({t.role.capitalize()}): {t.message}" for t in turns[-200:]])
    print("conversation id",conversation_id)
    print("turns",turns)
    print("dialog histoy",dialog_history)
    print("req file:",req.files)
    # 处理多个文件
    retrieved_docs_str = ''
    images_str = ''
    image_messages = []
    if req.files:
        all_documents = []
        all_images = []
        images_type = []
        all_audios = []
        for file_info in req.files:        
            file_path = file_info.get('path')
            print("PATH:",file_path)
            print("file_info:",file_info,type(file_info))
            print('##',file_info['filename'])
            # 根据 file_type 来拼接template (text,image,audio)
            try:
                if file_info.get('file_type') == "text":
                    documents = load_file_content(file_path,bucket)#本地
                    all_documents.extend(documents)
                    # print("All documents:",all_documents)
                elif file_info.get('file_type') == "image":
                    # image = encode_image(file_path)                
                    # all_images.append(image)  # 如果返回单个元素，则使用 append
                    # print("all image:",all_images)
                    image = bucket.sign_url('GET', file_path, 3600,params={'response-content-disposition': 'inline'})
                    image = f"{image}&t={int(time.time())}"
                    all_images.append(image)
                    images_type.append(file_info.get('filename').split('.')[-1].lower())
                    if len(all_images) != len(images_type):
                        raise ValueError(f"The number of images ({len(all_images)}) does not match the number of types ({len(images_type)})")
                elif file_info.get('file_type')== "audio":
                    pass
            except HTTPException as e:
                raise e
        # print("all documents:",all_documents)
        # print("all documents key:",all_documents[0].key())
        context = ""
        if all_documents:
            context += "\n\n\n".join([page.page_content for page in all_documents])
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            all_splits = text_splitter.split_documents(all_documents)
            _ = vector_store.add_documents(documents=all_splits)
            retrieved_docs = vector_store.similarity_search(req.message)
            # print('### retrieved docs',retrieved_docs[:3])
        retrieved_docs_str = f'-- text: {retrieved_docs[:3]}' if len(all_documents) > 0 else ''
        
            # images_str = '-- image_url:\n' + '\n'.join([f'data:image/{typ}; url:{img}' for typ, img in zip(images_type, all_images)])
        image_messages = [
            {"type": "image_url", "image_url": {"url": img}} for img in all_images
        ]
        
    
    
    dialog_history_str = f'{dialog_history}' if len(dialog_history) > 0 else ''
    user_question = req.message if req.message else ''

    # current_turn_message = f"""
    # You are an AI assistant. Answer the user's question using the retrieved context and conversation history (if available). 
    # If the context is insufficient, indicate that politely.

    # Context: 
    # {retrieved_docs_str};
    # {images_str}

    # Conversation History:
    # {dialog_history_str};

    # User Question: {user_question};
    # Answer:
    # """
    current_turn_message = [
    {"role": "system", "content": "You are an AI assistant. Answer the user's question using the retrieved context and conversation history (if available). If the context is insufficient, indicate that politely."},
    {"role": "user", "content": [
        {"type": "text", "text": f"Context: {retrieved_docs_str}"},
        *([{"type": "text", "text": "Images:"}] + image_messages if image_messages else []),  # 这里直接插入图片数据
        {"type": "text", "text": f"Conversation History: {dialog_history_str}"},
        {"type": "text", "text": f"User Question: {user_question}"}
    ]}
    ]
        

    
    # response = llm.invoke(current_turn_message)

    # ai_reply = response.content
    # ai_reply = re.sub(r"\(ai\):\s*", "", text, flags=re.I)
    
     # 保存用户消息到数据库
    user_turn = ConversationTurn(
        conversation_id=conversation_id,
        turn_number=turn_number,
        role="user",
        message=req.message
    )
    db.add(user_turn)
    db.commit()

    async def stream_ai_response() -> AsyncGenerator[str, None]:
        ai_reply = ''
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=current_turn_message,
                stream=True,
            )
            for chunk in stream:
                clean_chunk = re.sub(r"\(ai\):\s*", "", chunk.choices[0].delta.content or "", flags=re.I)
                ai_reply += clean_chunk
                yield clean_chunk
            
            ai_turn = ConversationTurn(
                conversation_id = conversation_id,
                turn_number = turn_number + 1,
                role = "ai",
                message=ai_reply,
            )
            db.add(ai_turn)
            db.commit()
        except Exception as e:
            print(f"Error during streaming: {e}")
            yield f"\n[Error] AI streaming failed: {e}"
    sresponse = StreamingResponse(stream_ai_response(), media_type="text/event-stream")
    sresponse.headers["Access-Control-Allw-Origin"] = "X-Conversation-ID"
    sresponse.headers["X-Conversation-ID"] = str(conversation_id)
    print("Conversation ID in headers:", sresponse.headers["X-Conversation-ID"])
    return sresponse

@router.get("/users/refresh-url")
def refresh_temp_url(path: str, current_user: User = Depends(get_current_active_user)):
    if not path:
        raise HTTPException(status_code=400, detail="Invalid path")

    try:
        temp_url = bucket.sign_url('GET', path.lstrip("/"), 3600,params={'response-content-disposition': 'inline'})# + f"&t={int(time.time())}" 
        
        return {"tempourl": temp_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate URL: {str(e)}")
    
@router.post("/users/chatimage1", response_class=StreamingResponse)
def chatimage_endpoint1(req: ChatRequestImage, current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    # 查询用户是否存在（省略验证）

    # 如果 conversation_id 为空则新建会话
    if req.conversation_id is None:
        new_conversation = Conversation(user_id=current_user.id, thread_id="thread_"+str(current_user.id)+"_"+str(int(time.time())))
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        conversation_id = new_conversation.id
    else:
        conversation_id = req.conversation_id

    turn_number = db.query(ConversationTurn).filter_by(conversation_id=conversation_id).count() + 1
    
    turns = db.query(ConversationTurn).filter_by(conversation_id=conversation_id).order_by(ConversationTurn.created_at.asc()).all() 
    print("TURNS",turns)
    for t in turns:
        print(f"ID: {t.id}, Role: {t.role}, Message: {t.message}, State Snapshot: {t.state_snapshot}")
    # dialog_history = [
    # {
    #     f'{t.role.capitalize()}': {
    #         'content': [
    #             {'type': 'text', 'text': f'{t.message}'}
    #         ] 
    #         # + (
    #         #     [{'type': 'image_url', 'image_url': {'url': f'{bucket.sign_url("GET", t.state_snapshot.get("path"), 3600,params={"response-content-disposition": "inline"})}'}}]
    #         #     if t.state_snapshot is not None and t.state_snapshot.get("path")
    #         #     else []
    #         # )
    #     }
    # }
    # for t in turns[-10:]
    # ]

    dialog_history = "\n".join([
    f"({t.role.capitalize()}): {t.message}"
    for t in turns[-10:] if getattr(t, "type", None) != "Image Gen"
    ])

    print('dialog_history',dialog_history)
    
    user_question = req.message if req.message else ''

   
    current_turn_message = [
    {"role": "system", "content": "\
     You are an AI summarizer tasked with generating a detailed textual description for use as input in DALL·E 3. \
     Based on the available conversation history, \
     and iterative adjustments—construct a precise and coherent prompt that accurately describes the desired image. \
     Ensure that any modifications or refinements from previous interactions are incorporated to align with the user's intended vision."},
    {"role": "user", "content": [
        {"type": "text", "text": f"Conversation History: {dialog_history}"},
        # *[
        #     {"type": "text", "text": turn.get('content', [{}])[0].get('text', '')}
        #     if isinstance(turn, dict) and turn.get('User') 
        #     else (
        #         {"type": "image_url", "image_url": {"url": turn.get('Ai', {}).get('content', [{}])[1].get('image_url', '')}}
        #         if len(turn.get('Ai', {}).get('content', [])) > 1  # 确保 content 至少有两个元素
        #         else {"type": "text", "text": turn.get('Ai', {}).get('content', [{}])[0].get('text', '')}  # 只有一个元素时，返回文本
        #     )
        #     for turn in dialog_history
        # ],
        {"type": "text", "text": f"User Question: {user_question}"}
    ]}
    ]

    user_turn = ConversationTurn(
        conversation_id=conversation_id,
        turn_number=turn_number,
        role="user",
        message=user_question
    )
    db.add(user_turn)
    db.commit()
    print('CHATIMAGE1:',req.message)
    async def stream_ai_response() -> AsyncGenerator[str, None]:
        ai_reply = ''
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=current_turn_message,
                stream=True,
            )
            for chunk in stream:
                clean_chunk = re.sub(r"\(ai\):\s*", "", chunk.choices[0].delta.content or "", flags=re.I)
                ai_reply += clean_chunk
                yield clean_chunk
            
            ai_turn = ConversationTurn(
                conversation_id = conversation_id,
                turn_number = turn_number + 1,
                role = "ai",
                message=ai_reply,
            )
            db.add(ai_turn)
            db.commit()
        except Exception as e:
            print(f"Error during streaming: {e}")
            yield f"\n[Error] AI streaming failed: {e}"
    sresponse = StreamingResponse(stream_ai_response(), media_type="text/event-stream")
    sresponse.headers["Access-Control-Allw-Origin"] = "X-Conversation-ID"
    sresponse.headers["X-Conversation-ID"] = str(conversation_id)
    print("Conversation ID in headers:", sresponse.headers["X-Conversation-ID"])
    return sresponse

@router.post("/users/chatimage2", response_model=ChatResponseImage)
def chatimage_endpoint2(req:ChatRequestImage, current_user: User = Depends(get_current_active_user),db: Session = Depends(get_db)):
    if req.conversation_id is None:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    conversation_id = req.conversation_id
    print('Chatimage2:',req.message)
    turn_number = db.query(ConversationTurn).filter_by(conversation_id=conversation_id).count() + 1

    if not isinstance(req.message, str):
        raise HTTPException(status_code=400, detail="message must be a string")
      
    try: 
        response = client.images.generate(
            model="dall-e-3",
            prompt=req.message,
            size="1024x1024",
            quality = "standard",
            n=1
        )
        # 如果 req.conversation_id 为 None，你直接 return，但FastAPI 需要明确返回 Response 或 JSONResponse，否则会返回 None，导致前端解析失败
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="DALL-E did not return any images")

        ai_reply = response.data[0].revised_prompt 
        modelurl = response.data[0].url
        print('后端Response',response)

        ##
        # 下载文件
        try:
            response_img = requests.get(modelurl, stream=True, timeout=10)
            response_img.raise_for_status()  # 确保请求成功
            image_bytes = io.BytesIO(response_img.content)  # 读取图片数据到内存
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to download image: {str(e)}")
        filename = f"{current_user.id}_{int(time.time())}.jpg"
        oss_object_name = f"{UPLOAD_DIR}/{filename}"
        ##
        headers = {'Content-Type': 'image/jpg'}
        
        bucket.put_object(oss_object_name, image_bytes.getvalue(), headers=headers)
        tempourl = bucket.sign_url('GET', oss_object_name, 3600,params={'response-content-disposition': 'inline'}) 
        existing_turn = db.query(ConversationTurn).filter_by(
            conversation_id=conversation_id, 
            turn_number=turn_number
        ).first()

        if existing_turn:
            # 记录已存在 -> 进行更新
            print('记录存在。')
            existing_turn.state_snapshot = {
                "type": "Image Gen",
                "path": oss_object_name,
                # "tempourl": tempourl
            }
        else:
            # 记录不存在 -> 创建新记录
            print('记录不存在。')
            print("临时签名",tempourl)
            ai_turn = ConversationTurn(
                conversation_id=conversation_id,
                turn_number=turn_number,
                role="ai",
                message='see the picture below',
                state_snapshot={
                    "type": "Image Gen",
                    "path": oss_object_name,
                    # "tempourl": tempourl
                }
            )
            db.add(ai_turn)  # 仅在新建时添加

        db.commit()  # 提交更新

    except Exception as e:
        db.rollback()
        print(f"Error at image generation: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
    
    return ChatResponseImage(reply=ai_reply, conversation_id=conversation_id,tempourl=tempourl,path=oss_object_name,type="Image Gen")

@router.post("/users/chataudioinput")
async def chat_audio_input(audiofile:UploadFile = File(...), current_user: User = Depends(get_current_active_user)):
    print('##1',audiofile)
    
    try:
        audio_bytes = await audiofile.read()
        audio_io = io.BytesIO(audio_bytes)#io.BytesIO 创建了一个“虚拟文件”，它在内存中模拟一个真实的文件对象。
        content_type = audiofile.content_type if audiofile.content_type else "audio/webm"

        # 发送到 OpenAI Whisper
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=("recording.webm", audio_io, content_type),
            response_format="json"
        )

        # transcription = client.audio.transcriptions.create(
        # model="whisper-1", 
        # file=audiofile.file)
        return {'text':transcription.text}
    except Exception as e:
        print('audio transcribing error:',e)
        return {'error':str(e)}
    
@router.post("/users/convert-audio")
def convert_audio(request: TextInput,current_user: User = Depends(get_current_active_user)):
    try:
        # 生成音频
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=request.text,
        )
        # 读取音频数据
        audio_bytes = response.content

        # 上传到 OSS
        filename = f"{current_user.id}_{int(time.time())}.mp3"
        oss_object_name = f"{UPLOAD_DIR_AUDIO}/{filename}"
        bucket.put_object(oss_object_name, audio_bytes, headers={'Content-Type': 'audio/mpeg'})

        # 生成签名 URL
        tempourl = bucket.sign_url('GET', oss_object_name, 3600, params={'response-content-disposition': 'inline'})


        return JSONResponse(content={"tempourl": tempourl,"path": oss_object_name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")


@router.post("/users/refresh-audio-url")
def refresh_audio_url(file_path: str):
    """
    仅刷新 OSS 签名 URL，不重新上传音频
    file_path 示例: "chat_audio/123456789.mp3"
    """
    try:
        # 生成新的签名 URL（1 小时有效期）
        tempourl = bucket.sign_url('GET', file_path, 3600, params={'response-content-disposition': 'inline'})
        return JSONResponse(content={"tempourl": tempourl})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh audio URL: {str(e)}")

@router.post("/users/delete-audio")
def delete_audio(request:DeleteAudioRequest):
    """
    删除 OSS 上的音频文件
    file_path 示例: "chat_audio/123456789.mp3"
    """
    file_path = request.file_path
    print("DELETE FILE PATH",file_path)
    try:
        # 检查文件是否存在
        if bucket.object_exists(file_path):
            bucket.delete_object(file_path)
            return JSONResponse(content={"message": "Audio deleted successfully."})
        else:
            return JSONResponse(content={"message": "File not found."}, status_code=404)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete audio: {str(e)}")
# [
# {'User': {'content': [{'type': 'text', 'text': 'cute goofy little cat on a snow and there is also an aircraft on the sky'}]}},
# {'Ai': {'content': [{'type': 'text', 'text': 'Create a whimsical scene featuring a cute, \
#                      goofy little cat playing in the snow. The cat should have a playful expression, \
#                      with its fur fluffed up and a cheerful demeanor, perhaps with snowflakes adorning its whiskers. In the sky above, \
#                      include a small aircraft flying by, leaving a trail of white smoke. The background should be a winter wonderland, with softly falling snow and a few snow-covered trees. \
#                      The overall mood is lighthearted and cheerful, capturing the playful spirit of the cat and the serene beauty of a snowy landscape.'}]}},
# {'Ai': {'content': [{'type': 'text', 'text': 'see the picture below'}, 
#                     {'type': 'image_url', 'image_url': {'url': 'https://pieces.oss-cn-beijing.aliyuncs.com/uploads%2Fimages-generation%2F%2F1_1742135862.jpg?response-content-disposition=inline&OSSAccessKeyId=LTAI5tJswmdZVcSMhavKouAD&Expires=1742139463&Signature=2q%2B2MJtXrY68Pk1%2Fq5RW5Dd2RVY%3D'}}]}}
# ]
    
