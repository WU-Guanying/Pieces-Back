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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=engine)