import dataclasses
import os as os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass, asdict
from typing import List
from typing import Dict
import json

import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from common.module import DATA_PATH


# === 1. 데이터 불러오기 (UTF-8 적용) ===
datStore = pd.read_csv(DATA_PATH / "dat_store.dat", sep='|', encoding='utf-8')
datSales = pd.read_csv(DATA_PATH / "dat_sales.dat", sep='|', encoding='utf-8')
datUj = pd.read_csv(DATA_PATH / "cd_uj.dat", sep='|', encoding='utf-8')
datDong = pd.read_csv(DATA_PATH / "cd_dong.dat", sep='|', encoding='utf-8')
datBrnd = pd.read_csv(DATA_PATH / "dat_brnd.dat", sep='|', encoding='utf-8')
datFchhq = pd.read_csv(DATA_PATH / "dat_fchhq.dat", sep='|', encoding='utf-8')
datGps = pd.read_csv(DATA_PATH / "cd_gps.dat", sep='|', encoding='utf-8')

# === 2. 해당 변수에 저장된 파일 CSV 변환(UTF-8 적용)
datStore.to_csv(DATA_PATH / "dat_store.csv", index=False, encoding='utf-8-sig')
datSales.to_csv(DATA_PATH / "dat_sales.csv", index=False, encoding='utf-8-sig')
datUj.to_csv(DATA_PATH / "cd_uj.csv", index=False, encoding='utf-8-sig')
datDong.to_csv(DATA_PATH / "cd_dong.csv", index=False, encoding='utf-8-sig')
datBrnd.to_csv(DATA_PATH / "dat_brnd.csv", index=False, encoding='utf-8-sig')
datFchhq.to_csv(DATA_PATH / "dat_fchhq.csv", index=False, encoding='utf-8-sig')
datGps.to_csv(DATA_PATH / "cd_gps.csv", index=False, encoding='utf-8-sig')