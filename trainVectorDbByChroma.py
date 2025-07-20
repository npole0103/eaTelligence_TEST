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
from langchain.document_loaders import TextLoader
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

from common.module import DATA_PATH, brandStatsVo, datStoreVo, datSalesVo, OUTPUT_PATH, LOGO_PATH, RESOURCE_PATH, CHUNK_SIZE

import warnings

warnings.filterwarnings(
	"ignore",
	category=DeprecationWarning,
	module=r"langchain_core\.tools",
)

load_dotenv()
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# PDF 로드
def load_pdfs(pdf_dir):
    docs = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    print(f"[INFO] 총 PDF 파일 수: {len(pdf_files)}")

    for filename in tqdm(pdf_files, desc="📄 PDF 로딩 중"):
        path = os.path.join(pdf_dir, filename)
        try:
            loader = PyPDFLoader(path)
            loaded = loader.load()
            docs.extend(loaded)
            print(f"[DEBUG] {filename} → 문서 {len(loaded)}개 로드")
        except Exception as e:
            print(f"[ERROR] {filename} 로드 실패: {e}")

    return docs

def load_txts(txt_dir):
    docs = []
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            path = os.path.join(txt_dir, filename)
            print(f"[DEBUG] 시도 중: {path}")
            try:
                loader = TextLoader(path, encoding="utf-8")
                doc = loader.load()
                print(f"[DEBUG] 로드 성공: {filename}, 문서 수: {len(doc)}")
                docs.extend(doc)
            except Exception as e:
                print(f"[ERROR] {filename} 로딩 실패: {e}")
    return docs

# 텍스트 분할 (RecursiveCharacterTextSplitter 사용)
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]  # 문단, 문장 단위 우선 분할
    )

    all_chunks = []
    for doc in tqdm(documents, desc="📚 문서 분할 중"):
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks

def brndStatsCsvToDocs(datBrnd, datFchhq, brandStats):
    docs = []
    brandStats = brandStats.merge(datBrnd[["brnd_nm", 'brnd_no', 'fchhq_no', 'uj3_nm']], on=['brnd_no', 'fchhq_no'], how='left')
    brandStats = brandStats.merge(datFchhq[["fchhq_nm", 'fchhq_no']], on='fchhq_no', how='left')

    ## float 컬럼 object로 캐스팅
    for col in brandStats.columns:
        if brandStats[col].dtype == "float64":
            brandStats[col] = brandStats[col].astype("object")

    brandStats = brandStats.where(pd.notnull(brandStats), None)
    brandStatsVoList: list[brandStatsVo] = [
        brandStatsVo(**row._asdict()) for row in brandStats.itertuples(index=False)
    ]

    for vo in brandStatsVoList:
        text = f"""
                {vo.ym_date} 기준, 프랜차이즈 {vo.fchhq_nm} 브랜드의 {vo.brnd_nm} 정보는 업종 소분류는 {vo.uj3_nm} 업종코드 {vo.uj3_cd}, 프랜차이즈 코드 {vo.fchhq_no},
                브랜드 코드 {vo.brnd_no}에 해당하는 브랜드의 개점 수는 {vo.store_open_cnt or 0}개,
                폐점 수는 {vo.store_close_cnt or 0}개이며 전체 점포 수는 {vo.store_cnt or 0}개입니다.
                NICE 기준 가맹점 수는 {vo.store_cnt_nice or 0}개이고,
                평균 매출은 {vo.amt_avg or 0}천원, 총 매출은 {vo.amt_total or 0}천원입니다.
                """
        doc = Document(
            page_content = re.sub(r"\s+", " ", text).strip(),
            metadata = {k: (v if v is not None else 0) for k, v in asdict(vo).items()}
        )
        docs.append(doc)
    return docs

def datStoreVoToDocs(datStore, cdDong, datBrnd, datFchhq, cdGps):
    docs = []

    ## 전처리
    datBrnd = datBrnd.rename(columns={"rprsv_nm": "brnd_rprsv_nm"}, inplace=False)
    datFchhq = datFchhq.rename(columns={"rprsv_nm": "fchhq_rprsv_nm"}, inplace=False)
    datBrnd = datBrnd.rename(columns={"y_ftc": "brnd_y_ftc"}, inplace=False)
    datFchhq = datFchhq.rename(columns={"y_ftc": "fchhq_y_ftc"}, inplace=False)
    cdDong = cdDong.drop(columns=['mega_nm', 'cty_nm']) # 칼럼 삭제
    datStore = datStore.dropna(subset=['fchhq_no', 'brnd_no']) # 값이 비었다면 행 삭제
    datStore = datStore.drop(columns=['uj3_cd']) # 칼럼 삭제
    datBrnd['ymd_brnd'] = datBrnd['ymd_brnd'].apply(
        lambda x: str(int(x)) if pd.notna(x) and isinstance(x, float) else str(x)
    )
    datStore["ym_start"] = datStore["ym_start"].apply(
        lambda x: str(int(x)) if pd.notna(x) and isinstance(x, float) else str(x)
    )

    datStore["ym_end"] = datStore["ym_end"].apply(
        lambda x: str(int(x)) if pd.notna(x) and isinstance(x, float) else str(x)
    )

    datStore = datStore.merge(cdGps, on="pnu", how="left")
    datStore = datStore.merge(cdDong, on="dong_cd", how="left")
    datStore = datStore.merge(datBrnd, on=['brnd_no', 'fchhq_no'], how="left")
    datStore = datStore.merge(datFchhq, on='fchhq_no', how="left")

    ## float 컬럼 object로 캐스팅
    for col in datStore.columns:
        if datStore[col].dtype == "float64":
            datStore[col] = datStore[col].astype("object")

    datStore = datStore.where(pd.notnull(datStore), None)
    allowed_keys = {f.name for f in fields(datStoreVo)}
    datStoreVoList: list[datStoreVo] = [
        datStoreVo(**{k: v for k, v in row._asdict().items() if k in allowed_keys}) for row in datStore.itertuples(index=False)
    ]

    for vo in datStoreVoList:
        text = f"""
            해당 가맹점은 {vo.ym_start}부터 운영을 시작했으며,
            위치는 {vo.zone_nm}에 속하고, 읍면동 코드는 {vo.dong_cd}입니다.
            
            지리적으로는 위도 {vo.gps_lat}°, 경도 {vo.gps_lon}°에 위치해 있으며,
            필지고유번호는 {vo.pnu}입니다.
            
            이 가맹점은 {vo.fchhq_nm} 프랜차이즈 소속으로, 프랜차이즈 코드는 {vo.fchhq_no}, 대표자는 {vo.fchhq_rprsv_nm or '정보없음'}입니다.
            해당 프랜차이즈의 브랜드는 {vo.brnd_nm}(브랜드 코드: {vo.brnd_no})이며, 브랜드 대표는 {vo.brnd_rprsv_nm or '정보없음'}입니다.
            
            브랜드는 {vo.ymd_brnd}에 가맹사업을 개시했으며,
            업종 소분류는 {vo.uj3_nm}(업종 코드: {vo.uj3_cd})에 해당합니다.
            
            {'해당 지점은 현재까지 운영 중인 가맹점입니다.' if vo.ym_end == '202502' else f'해당 지점 폐업일은 {vo.ym_end}입니다.'}
            """
        doc = Document(
            page_content = re.sub(r"\s+", " ", text).strip(),
            metadata = {k: (v if v is not None else 0) for k, v in asdict(vo).items()}
        )
        docs.append(doc)

    return docs

def datSalesVoToDocs(datSales, datBrnd, datFchhq, cdDong):
    docs = []
    datSales = datSales.merge(datBrnd[['brnd_nm', 'brnd_no', 'fchhq_no']], on=['brnd_no', 'fchhq_no'], how="left")
    datSales = datSales.merge(datFchhq[['fchhq_nm', 'fchhq_no']], on='fchhq_no', how="left")
    datSales = datSales.merge(cdDong, on='dong_cd', how="left")

    ## float 컬럼 object로 캐스팅
    for col in datSales.columns:
        if datSales[col].dtype == "float64":
            datSales[col] = datSales[col].astype("object")

    datSales = datSales.where(pd.notnull(datSales), None)
    allowed_keys = {f.name for f in fields(datSalesVo)}
    datSalesVoList: list[datSalesVo] = [
        datSalesVo(**{k: v for k, v in row._asdict().items() if k in allowed_keys}) for row in datSales.itertuples(index=False)
    ]

    for vo in datSalesVoList:
        zone_text = format_region_stats(
            prefix="읍면동",
            name=vo.zone_nm,
            cnt=vo.zone_cnt,
            avg=vo.zone_amt_avg,
            p25=vo.zone_amt_25pct,
            p50=vo.zone_amt_50pct,
            p75=vo.zone_amt_75pct,
            percnt=vo.zone_amt_percnt
        )

        cty_text = format_region_stats(
            prefix="시군구",
            name=vo.cty_nm,
            cnt=vo.cty_cnt,
            avg=vo.cty_amt_avg,
            p25=vo.cty_amt_25pct,
            p50=vo.cty_amt_50pct,
            p75=vo.cty_amt_75pct,
            percnt=vo.cty_amt_percnt
        )

        mega_text = format_region_stats(
            prefix="광역시도",
            name=vo.mega_nm,
            cnt=vo.mega_cnt,
            avg=vo.mega_amt_avg,
            p25=vo.mega_amt_25pct,
            p50=vo.mega_amt_50pct,
            p75=vo.mega_amt_75pct,
            percnt=vo.mega_amt_percnt
        )

        all_text = format_region_stats(
            prefix="전국",
            name="전체",
            cnt=vo.all_cnt,
            avg=vo.all_amt_avg,
            p25=vo.all_amt_25pct,
            p50=vo.all_amt_50pct,
            p75=vo.all_amt_75pct,
            percnt=vo.all_amt_percnt
        )

        # 본사/브랜드 정보가 있는지 확인
        has_brand = vo.fchhq_no is not None and vo.brnd_no is not None

        # 분기 처리
        if has_brand:
            header_text = f"""
        {vo.ym_sales} 기준, 본사 {vo.fchhq_nm} (ID: {vo.fchhq_no})와 브랜드 {vo.brnd_nm} (ID: {vo.brnd_no})는
        업종 소분류 '{vo.uj3_nm}' (코드: {vo.uj3_cd})에 속합니다.
        """
        else:
            header_text = f"""
        {vo.ym_sales} 기준, 업종 소분류 '{vo.uj3_nm}' (코드: {vo.uj3_cd})에 속하는 전체 가맹점 및 브랜드를 포함한 매출 통계입니다.
        특정 본사 또는 브랜드와는 연결되지 않은 일반 가맹점 데이터도 포함됩니다.
        """

        # 전체 문장 조립
        full_text = f"""
        {header_text}
        {zone_text}{cty_text}{mega_text}{all_text}
        """

        # Document 생성
        doc = Document(
            page_content=re.sub(r"\s+", " ", full_text).strip(),
            metadata={k: (v if v is not None else 0) for k, v in asdict(vo).items()}
        )
        docs.append(doc)

        # text = f"""
        # {vo.ym_sales} 기준, 본사 {vo.fchhq_nm} (ID: {vo.fchhq_no})와 브랜드 {vo.brnd_nm} (ID: {vo.brnd_no})는
        # 업종 소분류 '{vo.uj3_nm}' (코드: {vo.uj3_cd})에 속합니다.
        #
        # {zone_text}{cty_text}{mega_text}{all_text}
        # """
        # doc = Document(
        #     page_content=re.sub(r"\s+", " ", text).strip(),
        #     metadata={k: (v if v is not None else 0) for k, v in asdict(vo).items()}
        # )
        # docs.append(doc)

    return docs

def format_region_stats(prefix: str, name: str, cnt, avg, p25, p50, p75, percnt) -> str:
    if cnt is None or cnt < 5:
        return ""

    lines = [f"- {prefix}({name}) 단위에서는 총 {int(cnt)}개 가맹점이 운영 중이며,"]

    if cnt >= 5:
        lines.append(f"  평균 매출은 {int(avg) if avg is not None else 0}천원,")
        lines.append(f"  건당 평균 매출은 {int(percnt) if percnt is not None else 0}천원입니다.")

    if cnt >= 10:
        lines.insert(-1, f"  상위 25%는 {int(p25) if p25 is not None else 0}천원,")
        lines.insert(-1, f"  상위 50%는 {int(p50) if p50 is not None else 0}천원,")
        lines.insert(-1, f"  상위 75%는 {int(p75) if p75 is not None else 0}천원,")

    return "\n".join(lines) + "\n"

# CSV 데이터 가공 및 로드
def load_csvs():
    docs = []

    logging.info("엑셀 데이터 로드..")
    brandStats = pd.read_csv(DATA_PATH / "brand_stats_v2.csv", encoding='utf-8')
    datStore = pd.read_csv(DATA_PATH / "dat_store_v2.csv", encoding='utf-8')
    datSales = pd.read_csv(DATA_PATH / "dat_sales_v3.csv", encoding='utf-8')

    cdDong = pd.read_csv(DATA_PATH / "cd_dong_v2.csv", encoding='utf-8')
    datBrnd = pd.read_csv(DATA_PATH / "dat_brnd_v2.csv", encoding='utf-8')
    datFchhq = pd.read_csv(DATA_PATH / "dat_fchhq_v2.csv", encoding='utf-8')
    cdGps = pd.read_csv(DATA_PATH / "cd_gps_v2.csv", encoding='utf-8')

    # DAT_STORE 가맹점 정보 테이블 DOCS
    logging.info("DAT_STORE 가맹점 정보 테이블 DOCS 처리중..")
    docs.extend(datStoreVoToDocs(datStore, cdDong, datBrnd, datFchhq, cdGps))

    # BRAND_STATS 브랜드 통계 테이블 DOCS
    logging.info("BRAND_STATS 브랜드 통계 테이블 DOCS 처리중..")
    docs.extend(brndStatsCsvToDocs(datBrnd, datFchhq, brandStats))

    # DAT_SALES 매출 통계 테이블 DOCS
    logging.info("DAT_SALES 매출 통계 테이블 DOCS 처리중..")
    docs.extend(datSalesVoToDocs(datSales, datBrnd, datFchhq, cdDong))

    del brandStats
    del datStore
    del datSales

    del cdDong
    del datBrnd
    del datFchhq
    del cdGps

    gc.collect()

    return docs

# 4. 전체 처리 파이프라인
def build_chroma_vectorstore(pdf_dir, txt_dir, persist_dir):
    # (1) 로딩 데이터 셋팅
    logging.info("MAIN 데이터 로드 시작..")
    documents = load_pdfs(pdf_dir)
    documents.extend(load_txts(txt_dir))
    # documents = load_csvs()

    print(f"[DEBUG] 문서 수: {len(documents)}")

    # (2) 텍스트 분할
    logging.info("MAIN 텍스트 분할 SPLIT")
    split_docs = split_documents(documents)

    # (3) OpenAI 임베딩
    logging.info("OpenAI 임베딩 시작")
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=CHATGPT_API_KEY,
        chunk_size=100  # 배치 사이즈 이슈로 옵션 추가
    )

    # (4) Chroma DB 경로 존재 여부 확인
    total = len(split_docs)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logging.info("기존 Chroma DB 감지, 벡터 추가 중...")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

        for i in tqdm(range(0, total, CHUNK_SIZE)):
            chunk = split_docs[i:i + CHUNK_SIZE]
            if chunk:  # 안전 확인
                vectordb.add_documents(chunk)

    else:
        logging.info("신규 Chroma DB 생성 중...")

        if total == 0:
            raise ValueError("❌ split_docs가 비어 있어 Chroma를 생성할 수 없습니다.")

        # 첫 배치로 초기 생성
        first_batch = split_docs[:CHUNK_SIZE]
        vectordb = Chroma.from_documents(
            documents=first_batch,
            embedding=embedding,
            persist_directory=persist_dir
        )

        # 남은 배치 추가
        for i in tqdm(range(CHUNK_SIZE, total, CHUNK_SIZE)):
            chunk = split_docs[i:i + CHUNK_SIZE]
            if chunk:  # 방어적 처리
                vectordb.add_documents(chunk)

    logging.info(f"✅ 총 {len(split_docs)} chunks 저장 완료!")


# 실행
if __name__ == "__main__":
    build_chroma_vectorstore(pdf_dir="./pdf", txt_dir="./txt",persist_dir="./chroma_db")