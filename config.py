import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = BASE_DIR / "storage"

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')


