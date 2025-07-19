import dataclasses
import logging
import os
from pathlib import Path
import re
import gc
import math
from string import Template
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
from pathlib import Path
import time

from langchain.agents import initialize_agent, AgentType
from tqdm import tqdm
from typing import Optional

import openai
from click.core import batch
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass, asdict, fields
from typing import List, Dict
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
import shutil

import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.tools import Tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.google_serper import GoogleSerperRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from common.module import DATA_PATH, brandStatsVo, datStoreVo, datSalesVo, OUTPUT_PATH, LOGO_PATH, RESOURCE_PATH

import warnings

warnings.filterwarnings(
	"ignore",
	category=DeprecationWarning,
	module=r"langchain_core\.tools",
)

load_dotenv()
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# 데이터 전처리


# --------------------------
# 1️⃣ 데이터 로드 (예시)
# 실제로는 dat_sales, dat_store, dat_brnd 등을 병합해야 합니다
dat_sales = pd.read_csv("dat_sales.csv")  # 매출 데이터
# dat_brnd, dat_store 등 필요시 merge

# --------------------------
# 2️⃣ 라벨 생성 (전국 상위 25% 기준)
dat_sales['label'] = (dat_sales['amt_avg'] >= dat_sales['natl_amt_25pct']).astype(int)

# --------------------------
# 3️⃣ 피처셋 준비
features = [
    'store_cnt', 'store_area_m2', 'brnd_ads_m2', 'brnd_strt_ym',
    'zone_nm', 'cty_nm', 'mega_nm', 'submit_yn', 'gps_x', 'gps_y'
]

X = dat_sales[features].copy()
y = dat_sales['label']

# 카테고리형 지정
for col in ['zone_nm', 'cty_nm', 'mega_nm', 'submit_yn']:
    X[col] = X[col].astype('category')

# --------------------------
# 4️⃣ 학습/검증 데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# 5️⃣ LightGBM 학습
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=500,
    early_stopping_rounds=50
)

# --------------------------
# 6️⃣ 검증 데이터 예측
y_pred_proba = model.predict(X_valid)
y_pred = (y_pred_proba >= 0.5).astype(int)

# --------------------------
# 7️⃣ 성능 평가
print(f"AUC: {roc_auc_score(y_valid, y_pred_proba):.4f}")
print("Classification report:")
print(classification_report(y_valid, y_pred))

# --------------------------
# 8️⃣ 신규 데이터 예측 (X_new = 신규 점포 피처셋)
# preds = model.predict(X_new)
# 결과 해석: preds 값이 0~1 사이 확률, 1에 가까울수록 유망할 가능성 높음
