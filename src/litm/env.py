from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")
