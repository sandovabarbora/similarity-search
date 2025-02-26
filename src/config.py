from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Storage
    UPLOAD_FOLDER: str = "data/uploads"
    VECTOR_DIMENSION: int = 512  # Feature vector dimension

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Processing
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
