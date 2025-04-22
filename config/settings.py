from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # directories
    CONTEXT_DIR: str = Field("context", env="ENH_RAG_CONTEXT_DIR")

    # embedding
    EMBEDDING_MODEL: str = Field(..., env="EMB_MODEL_NAME")

    # vector db
    VECTOR_DB: str = Field("faiss", env="VECTOR_DB")
    VECTOR_INDEX_DIR: str = Field("faiss_indexes", env="VECTOR_INDEX_DIR")

    # LLM
    LLM_MODEL: str = Field(..., env="LLM_MODEL_NAME")
    LLM_TEMPERATURE: float = Field(0.2, env="LLM_TEMPERATURE")
    MAX_OUTPUT_TOKENS: int = Field(512, env="MAX_OUTPUT_TOKENS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
