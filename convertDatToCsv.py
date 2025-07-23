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

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.schema import Document
# from langchain_core.tools import Tool
# from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
# from langchain_community.tools.google_serper import GoogleSerperRun
# from langchain.utilities import GoogleSerperAPIWrapper
# from langchain_community.tools.playwright.utils import create_async_playwright_browser
# from langchain_community.tools.playwright.utils import create_sync_playwright_browser
# from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
# from langgraph.graph import StateGraph, END
# from langchain_openai import ChatOpenAI

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

# # 25.07.19(토) DAT_BRND 테이블 전화번호 업데이트 V3
#
# datBrnd = pd.read_csv(DATA_PATH / "dat_brnd_v3.csv", encoding='utf-8-sig')
# datBrnd.to_csv(DATA_PATH / "dat_brnd_v3.dat", sep="|", index=False, encoding="utf-8")

# # 25.07.21(월) DAT_STORE V3 생성(geo_cd) / dat_keyword_search_v3.csv / dat_uj_device_v3.csv 추가
#
# datStore = pd.read_csv(DATA_PATH / "dat_store_v2.dat", sep='|', encoding='utf-8')
# statDongMapping = pd.read_csv(DATA_PATH / "stat_dong_mapping.csv", encoding='cp949')
#
# datStore = datStore.merge(statDongMapping[['dong_cd', 'geo_cd']], on='dong_cd', how='left')
#
# datStore.to_csv(DATA_PATH / "dat_store_v3.csv", index=False, encoding='utf-8-sig')
# datStore.to_csv(DATA_PATH / "dat_store_v3.dat", sep="|", index=False, encoding="utf-8")
#
# datKeywordSearch = pd.read_csv(DATA_PATH / "dat_keyword_search_v3.csv", encoding='cp949')
# datUjDevice = pd.read_csv(DATA_PATH / "dat_uj_device_v3.csv", encoding='cp949')
#
# datKeywordSearch.to_csv(DATA_PATH / "dat_keyword_search_v3.dat", sep='|', index=False, encoding='utf-8')
# datUjDevice.to_csv(DATA_PATH / "dat_uj_device_v3.dat", sep='|', index=False, encoding='utf-8')

# # 25.07.23(수) 컨버팅
# statDongMapping = pd.read_csv(DATA_PATH / "stat_dong_mapping.csv", encoding='cp949')
# statDongMapping.to_csv(DATA_PATH / "stat_dong_mapping_v3.dat", sep='|', index=False, encoding='utf-8')
# statDongMapping.to_csv(DATA_PATH / "stat_dong_mapping_v3.csv", index=False, encoding='utf-8-sig')
#
# datKeywordSearch = pd.read_csv(DATA_PATH / "dat_keyword_search_v3.dat", sep='|', encoding='utf-8')
# datKeywordSearch.to_csv(DATA_PATH / "dat_keyword_search_v3.csv", index=False, encoding='utf-8-sig')
#
# datUjDevice = pd.read_csv(DATA_PATH / "dat_uj_device_v3.dat", sep='|', encoding='utf-8')
# datUjDevice.to_csv(DATA_PATH / "dat_uj_device_v3.csv", index=False, encoding='utf-8-sig')

# # 25.07.23(수) 재변환
# datKeywordSearch = pd.read_csv(DATA_PATH / "dat_keyword_search_v3.csv", encoding='utf-8')
# datKeywordSearch.to_csv(DATA_PATH / "dat_keyword_search_v3.dat", index=False, sep='|', encoding='utf-8')
#
# datUjDevice = pd.read_csv(DATA_PATH / "dat_uj_device_v3.csv", encoding='utf-8')
# datUjDevice.to_csv(DATA_PATH / "dat_uj_device_v3.dat", index=False, encoding='utf-8-sig')

# 25.07.23(수) ML 버전 변환
brandStat = pd.read_csv(DATA_PATH / "brand_stat_v4.csv", encoding='utf-8')
datSalesAppend = pd.read_csv(DATA_PATH / "dat_sales_append_v4.csv", encoding='utf-8')

brandStat.to_csv(DATA_PATH / "brand_stat_v4.dat", index=False, encoding='utf-8-sig')
datSalesAppend.to_csv(DATA_PATH / "dat_sales_append_v4.dat", index=False, encoding='utf-8-sig')