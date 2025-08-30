import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "PDF Reconstruction Tool"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Keys
    GEMINI_API_KEY: str = ""
    AZURE_VISION_KEY: str = ""
    AZURE_VISION_ENDPOINT: str = ""
    
    # File settings
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf"}
    
    # Processing settings
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    SIMILARITY_THRESHOLD: float = 0.3
    
    # CORS settings
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create directories
def create_directories():
    upload_dir = Path(Settings().UPLOAD_DIR)
    output_dir = Path(Settings().OUTPUT_DIR)
    upload_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

settings = Settings()
create_directories()