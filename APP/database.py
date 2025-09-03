# database.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from .models import Base

# DATABASE_URL = "sqlite:///./test.db"  # You can use any database here
DATABASE_URL = "mysql+pymysql://wuguanying:Biosphere_00@rm-bp18lhp5wc42y150sro.mysql.rds.aliyuncs.com:3306/PIECES"
# DATABASE_URL = "mysql+pymysql://root:qaz287335728@127.0.0.1:3306/PIECES"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()
# 创建数据库引擎（engine），连接数据库。
# 设置数据库会话类 SessionLocal，用来在每个请求中创建和关闭数据库连接。
# 创建一个空的 MetaData() 对象，一般在你自己管理表结构时用。
# 这是一个依赖注入函数，常见于 FastAPI，用于获取数据库会话并在使用后自动关闭。
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 根据你定义的 ORM 模型类自动在数据库中创建表
Base.metadata.create_all(bind=engine)
