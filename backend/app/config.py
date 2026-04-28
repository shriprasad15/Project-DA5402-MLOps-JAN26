from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    BACKEND_ADMIN_TOKEN: str = "dev-token"
    MODEL_SERVER_URL: str = "http://model-server:8080"
    POSTGRES_USER: str = "mlops"
    POSTGRES_PASSWORD: str = "mlops"
    POSTGRES_DB: str = "mlops"
    POSTGRES_HOST: str = ""  # empty = use SQLite
    MODEL_CLIENT: Literal["mock", "http"] = "mock"
    LOG_LEVEL: str = "INFO"
    MAILTRAP_HOST: str = "sandbox.smtp.mailtrap.io"
    MAILTRAP_PORT: int = 2525
    MAILTRAP_USER: str = ""          # also accepted as MAILTRAP_USERNAME
    MAILTRAP_USERNAME: str = ""
    MAILTRAP_PASS: str = ""          # also accepted as MAILTRAP_PASSWORD
    MAILTRAP_PASSWORD: str = ""
    MAILTRAP_FROM: str = "pa-detector@example.com"
    MAILTRAP_TO: str = "admin@example.com"
    MAILTRAP_TO_EMAIL: str = ""

    @property
    def mailtrap_user(self) -> str:
        return self.MAILTRAP_USER or self.MAILTRAP_USERNAME

    @property
    def mailtrap_pass(self) -> str:
        return self.MAILTRAP_PASS or self.MAILTRAP_PASSWORD

    @property
    def mailtrap_to(self) -> str:
        return self.MAILTRAP_TO_EMAIL or self.MAILTRAP_TO


def get_settings() -> Settings:
    return Settings()
