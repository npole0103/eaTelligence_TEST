import dataclasses
import logging
import os as os
from pathlib import Path
import re
import gc
from tqdm import tqdm

import openai
from click.core import batch
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass, asdict, fields
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
from langchain.schema import Document

from common.module import DATA_PATH, brandStatsVo, datStoreVo, datSalesVo


load_dotenv()
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")

# # === 1. 데이터 불러오기 (UTF-8 적용) ===
# datStore = pd.read_csv(DATA_PATH / "dat_store.dat", sep='|', encoding='utf-8')
# datSales = pd.read_csv(DATA_PATH / "dat_sales.dat", sep='|', encoding='utf-8')
# datUj = pd.read_csv(DATA_PATH / "cd_uj.dat", sep='|', encoding='utf-8')
# datDong = pd.read_csv(DATA_PATH / "cd_dong.dat", sep='|', encoding='utf-8')
# datBrnd = pd.read_csv(DATA_PATH / "dat_brnd.dat", sep='|', encoding='utf-8')
# datFchhq = pd.read_csv(DATA_PATH / "dat_fchhq.dat", sep='|', encoding='utf-8')
# datGps = pd.read_csv(DATA_PATH / "cd_gps.dat", sep='|', encoding='utf-8')
#
# # === 2. 해당 변수에 저장된 파일 CSV 변환(UTF-8 적용)
# datStore.to_csv(DATA_PATH / "dat_store.csv", index=False, encoding='utf-8-sig')
# datSales.to_csv(DATA_PATH / "dat_sales.csv", index=False, encoding='utf-8-sig')
# datUj.to_csv(DATA_PATH / "cd_uj.csv", index=False, encoding='utf-8-sig')
# datDong.to_csv(DATA_PATH / "cd_dong.csv", index=False, encoding='utf-8-sig')
# datBrnd.to_csv(DATA_PATH / "dat_brnd.csv", index=False, encoding='utf-8-sig')
# datFchhq.to_csv(DATA_PATH / "dat_fchhq.csv", index=False, encoding='utf-8-sig')
# datGps.to_csv(DATA_PATH / "cd_gps.csv", index=False, encoding='utf-8-sig')

# # 25.O7.09 V2 가공
# # === 1. 데이터 불러오기 (UTF-8 적용) ===
# datStore = pd.read_csv(DATA_PATH / "dat_store_v2.dat", sep='|', encoding='utf-8')
# datSales = pd.read_csv(DATA_PATH / "dat_sales_v2.dat", sep='|', encoding='utf-8')
# datDong = pd.read_csv(DATA_PATH / "cd_dong_v2.dat", sep='|', encoding='utf-8')
# datBrnd = pd.read_csv(DATA_PATH / "dat_brnd_v2.dat", sep='|', encoding='utf-8')
# datFchhq = pd.read_csv(DATA_PATH / "dat_fchhq_v2.dat", sep='|', encoding='utf-8')
# datGps = pd.read_csv(DATA_PATH / "cd_gps_v2.dat", sep='|', encoding='utf-8')
#
# # === 2. 해당 변수에 저장된 파일 CSV 변환(UTF-8 적용)
# datStore.to_csv(DATA_PATH / "dat_store_v2.csv", index=False, encoding='utf-8-sig')
# datSales.to_csv(DATA_PATH / "dat_sales_v2.csv", index=False, encoding='utf-8-sig')
# datDong.to_csv(DATA_PATH / "cd_dong_v2.csv", index=False, encoding='utf-8-sig')
# datBrnd.to_csv(DATA_PATH / "dat_brnd_v2.csv", index=False, encoding='utf-8-sig')
# datFchhq.to_csv(DATA_PATH / "dat_fchhq_v2.csv", index=False, encoding='utf-8-sig')
# datGps.to_csv(DATA_PATH / "cd_gps_v2.csv", index=False, encoding='utf-8-sig')

# # 25.07.12 STATS 통계 데이터 가공
# brandStats = pd.read_csv(DATA_PATH / "brand_stats_v2.csv", encoding='utf-8-sig')
# brandStats["store_cnt_nice"] = brandStats["store_cnt_nice"].apply(lambda x: int(x) if pd.notna(x) else "")
# brandStats["amt_avg"] = brandStats["amt_avg"].apply(lambda x: int(x) if pd.notna(x) else "")
# brandStats["amt_total"] = brandStats["amt_total"].apply(lambda x: int(x) if pd.notna(x) else "")
# brandStats.to_csv(DATA_PATH / "dat_stats_v2.dat", sep="|", index=False, encoding="utf-8")

# # 25.07.12 STATS 통계 데이터 검증
# datStore = pd.read_csv(DATA_PATH / "dat_store_v2.dat", sep='|', encoding='utf-8')
# cnt = datStore[(datStore["brnd_no"] == "BRD_20100100504")
#                & (datStore["ym_end"] > 202311)
#                & (datStore["ym_start"] <= 202311)].shape[0]
# print(cnt)

# # 25.07.13 DAT_SALES 테이블 uj3_nm 필드 칼럼 추가
# datSales = pd.read_csv(DATA_PATH / "dat_sales.dat", sep='|', encoding='utf-8')
# datUj = pd.read_csv(DATA_PATH / "cd_uj.dat", sep='|', encoding='utf-8')
#
# datSales = datSales.merge(datUj[['uj3_cd', 'uj3_nm']], on='uj3_cd', how="left")
#
# datSales.to_csv(DATA_PATH / "dat_sales_v3.dat", sep="|", index=False, encoding="utf-8")
# datSales.to_csv(DATA_PATH / "dat_sales_v3.csv", index=False, encoding='utf-8-sig')