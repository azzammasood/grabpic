from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://grabpic:grabpic@localhost:5432/grabpic"
    storage_path: str = "./storage"
    face_model: str = "Facenet512"
    detector_backend: str = "retinaface"
    # Cosine similarity threshold for matching (higher = stricter).
    match_threshold: float = 0.42
    api_title: str = "Grabpic"
    api_version: str = "1.0.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
