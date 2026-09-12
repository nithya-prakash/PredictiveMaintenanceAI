from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Predictive Maintenance AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "predictive_maintenance"
    POSTGRES_PORT: str = "5432"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
