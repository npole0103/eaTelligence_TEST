import dataclasses
import os as os
from pathlib import Path

import openai
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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from common.module import DATA_PATH

load_dotenv()
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")

# === 1. 데이터 불러오기 (UTF-8 적용) ===
datStore = pd.read_csv(DATA_PATH / "dat_store.dat", sep='|', encoding='utf-8')
datSales = pd.read_csv(DATA_PATH / "dat_sales.dat", sep='|', encoding='utf-8')
datUj = pd.read_csv(DATA_PATH / "cd_uj.dat", sep='|', encoding='utf-8')
datDong = pd.read_csv(DATA_PATH / "cd_dong.dat", sep='|', encoding='utf-8')
datBrnd = pd.read_csv(DATA_PATH / "dat_brnd.dat", sep='|', encoding='utf-8')
datFchhq = pd.read_csv(DATA_PATH / "dat_fchhq.dat", sep='|', encoding='utf-8')
datGps = pd.read_csv(DATA_PATH / "cd_gps.dat", sep='|', encoding='utf-8')

# === 2. 데이터 병합 ===
## 업종 코드 정보
datStore = datStore.merge(datUj, on='uj3_cd', how='left')
## 행정동 정보
datStore = datStore.merge(datDong, on='dong_cd', how='left')
## 브랜드 정보
datStore = datStore.merge(datBrnd[['brnd_nm', 'brnd_no', 'brnd_inds_nm1', 'brnd_inds_nm2', 'majr_gds', 'store_cnt']], on='brnd_no', how='left')
## 본사 정보
datStore = datStore.merge(datFchhq[['fchhq_no', 'fchhq_nm']], on='fchhq_no', how='left')
## GPS 좌표 정보
datStore = datStore.merge(datGps[['pnu', 'gps_lat', 'gps_lon']], on='pnu', how='left')
## 매출 정보 (업종 + 브랜드 + 행정동 기준)
datStore = datStore.merge(datSales, on=['uj3_cd', 'brnd_no', 'dong_cd'],how='left')

datStore.head(100).to_csv(DATA_PATH / "sample_5.csv", index=False, encoding='utf-8-sig')

# # === 3. 타겟 생성 (상위 30%) ===
# datStore = datStore.dropna(subset=['mall_amt_avg'])
# datStore['target'] = (datStore['mall_amt_avg'] >= datStore['mall_amt_avg'].quantile(0.7)).astype(int)
#
# # === 4. 입지 클러스터링 ===
# datStore = datStore.dropna(subset=['gps_lat', 'gps_lng'])
# kmeans = KMeans(n_clusters=10, random_state=42)
# datStore['location_cluster'] = kmeans.fit_predict(datStore[['gps_lat', 'gps_lng']])
#
# # === 5. 피처 준비 ===
# feature_cols = ['uj3_nm', 'uj2_nm', 'dong_nm', 'inds_nm1', 'inds_nm2', 'hq_nm', 'location_cluster']
# X = pd.get_dummies(datStore[feature_cols])
# y = datStore['target']
#
# # === 6. 모델 학습 및 평가 ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = lgb.LGBMClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# acc = accuracy_score(y_test, y_pred)
# print(f"✅ 모델 정확도: {acc:.4f}")
