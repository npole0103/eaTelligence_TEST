import pandas as pd
from src.config import db_config as config


def load_excel(file_key: str) -> pd.DataFrame:
    path = config.EXCEL_FILES[file_key]
    return pd.read_csv(path, encoding='utf-8')