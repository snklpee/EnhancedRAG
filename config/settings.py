# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from pathlib import Path

class Settings(BaseSettings):
    # directories
    CONTEXT_DIR: str = Field("context", env="CONTEXT_DIR")
    FAISS_INDEXES: str = ("context/faiss_indexes", "FAISS_INDEXES")
    
    # OPENAI_API_KEY: SecretStr = Field(..., env="OPENAI_API_KEY")
    HF_TOKEN: SecretStr = Field(..., env="HF_TOKEN")

    class Config:
        env_file = str(Path(__file__).parent / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
