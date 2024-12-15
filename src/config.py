import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_ARTIFACTS_DIR = os.path.join(PROCESSED_DATA_DIR, "model_artifacts")

# API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# 模型配置
MODEL_NAME = "microsoft/deberta-v3-base"  # 用于fine-tuning的基础模型
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3