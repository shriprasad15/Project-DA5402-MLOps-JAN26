from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base


def get_engine(
    postgres_host: str = "",
    postgres_user: str = "mlops",
    postgres_password: str = "mlops",
    postgres_db: str = "mlops",
):
    if postgres_host:
        url = (
            f"postgresql+psycopg2://{postgres_user}:{postgres_password}"
            f"@{postgres_host}/{postgres_db}"
        )
    else:
        url = "sqlite:///./pa_detector.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def get_db(engine):
    SessionLocal.configure(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
