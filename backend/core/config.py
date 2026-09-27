from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    PROJECT_NAME: str = "Predictive Maintenance AI"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    MODEL_PATH: str = "models/model_bundle.joblib"

    # Database. There is deliberately no default password: set POSTGRES_PASSWORD
    # in .env (docker compose refuses to start without it). Without a reachable
    # database the API still serves predictions; only persistence is skipped.
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "predictive_maintenance"
    POSTGRES_PORT: str = "5432"

    # Browser origins allowed to call the API (the dashboard calls it server-side).
    CORS_ORIGINS: List[str] = ["http://localhost:8501"]
    # Per-client-IP request limit on the prediction endpoints.
    RATE_LIMIT_PER_MINUTE: int = 120
    # Optional shared token for trusted internal callers (the dashboard), sent as
    # X-API-Key. They are exempt from the per-IP limit: every dashboard user reaches
    # the API from the one dashboard container, so limiting its IP throttles them all.
    API_TOKEN: str = ""

    @property
    def database_url(self) -> str:
        return (f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:"
                f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}")


settings = Settings()
