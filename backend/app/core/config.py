import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Snooker Rack CCTV Session Dashboard"
    API_V1_STR: str = "/api"
    
    # Database
    DATABASE_URL: str = "sqlite:///snooker_dashboard.db"
    
    # YOLO Model Configuration
    WEIGHTS_PATH: str = "runs/detect/snooker_rack_yolov12/weights/best.pt"
    FALLBACK_WEIGHTS: list = [
        "yolo12n.pt",
        "models/snooker_rack_yolov11.pt",
        "models/snooker_rack_yolov12.pt"
    ]
    CONFIDENCE_THRESHOLD: float = 0.50
    
    # Default billing rate per frame/game
    DEFAULT_GAME_RATE: float = 10.0

    class Config:
        case_sensitive = True

settings = Settings()

# Ensure we locate the best available weights
def get_model_weights():
    if os.path.exists(settings.WEIGHTS_PATH):
        return settings.WEIGHTS_PATH
    for w in settings.FALLBACK_WEIGHTS:
        if os.path.exists(w):
            return w
        elif os.path.exists(os.path.join("models", w)):
            return os.path.join("models", w)
    return "yolo12n.pt"
