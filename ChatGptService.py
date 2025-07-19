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

BATCH_SIZE = 5

client = OpenAI(api_key=CHATGPT_API_KEY)

@dataclass(frozen=False)
class FranchiseStore:
    franchise_name: str       # 프랜차이즈명
    brand_name: str           # 브랜드명
    store_count: int          # 가맹점수
    business_type: str        # 업종종류
    location: str             # 위치
    monthly_sales: Dict[str, float]  # 1~12월 월매출 (배열 12개)

@dataclass(frozen=False)
class brandVo:
    brnd_no: str
    brnd_nm: str
    tel_no: str
    source: Optional[str] = None

def generateBrand(brandName):
    ## 프롬포트 최초 분석 (o3 모델)
    messages = [
        {"role": "system",
         "content": """
         너는 지금부터 프랜차이즈와 브랜드를 분석하는 분석전문가야.
         그리고 내가 질문하는 모든 것에 대해 대답해줘야 하고 근거 있고 신뢰성 있는 데이터여야 해"""},

        {"role": "user", "content": f"""{brandName}에 대해서 
        프랜차이즈명, 브랜드명, 가맹점수, 업종종류, 위치(시군구동), 최근 12개월 각 매출 정보(천원단위)
        데이터 조사해줘"""}]

    firstRes = client.chat.completions.create(
        model="o3",
        messages=messages
    )

    ## 질의 데이터 가공 (4o 모델)
    initFranchiseData = firstRes.choices[0].message.content
    print("FIRST RES DATA", initFranchiseData)

    messages.extend([
    {"role": "assistant","content": f"""{initFranchiseData}"""},
    {"role": "system", "content": """
            당신은 항상 유효한 JSON만 반환하는 AI입니다. 설명, 주석, 추가 문장은 절대 포함하지 마세요.
            """},
    {"role": "user", "content": """
            방금 조사한 데이터를 다음과 같은 구조를 가진 JSON 객체를 반환해 주세요:
            {
              "franchise_name": "str",
              "brand_name": "str",
              "store_count": int,
              "business_type": "str",
              "location": "str",
              "monthly_sales": {
                  "1":float,
                  "2":float,
                  "3":float,
                  ..
                  "12":float
              }
            }
            예시 값:
            - franchise_name: 카페프렌즈
            - brand_name: 프렌즈커피
            - store_count: 120
            - business_type: 카페
            - location: 서울 강남구
            - monthly_sales: 1~12월 월매출 (배열 12개)
            
            반드시 JSON만 반환하세요.
            ```json ~ ``` 이런 블럭도 넣지마세요
            """}
    ])

    secondRes = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    ## 최종 데이터 가공
    franchiseDataByGpt = secondRes.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    print("GPT JSON DATA:", franchiseDataByGpt)

    try:
        data_dict = json.loads(franchiseDataByGpt)
    except json.JSONDecodeError as e:
        print("JSON 파싱 실패:", e)
        data_dict = None

    # 4️⃣ dataclass에 주입
    if data_dict:
        store = FranchiseStore(**data_dict)
        print("Dataclass 객체:", store)
    else:
        print("데이터 생성 실패")

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def generateTelNo():
    resBrandVoList: list [brandVo] = []

    ## LOAD DAT_BRND 데이터
    datBrnd = pd.read_csv(DATA_PATH / "dat_brnd_v2.csv", encoding='utf-8')

    datBrnd = datBrnd[['brnd_no', 'brnd_nm', 'tel_no', 'is_submit']]
    datBrnd = datBrnd[(datBrnd["is_submit"] == 1)]

    datBrnd["is_submit"] = datBrnd["is_submit"].apply(
        lambda x: int(x) if isinstance(x, float) and x.is_integer() else x)

    print(f"Data Cnt : {datBrnd.shape[0]}")

    datBrnd = datBrnd.where(pd.notnull(datBrnd), None)
    allowed_keys = {f.name for f in fields(brandVo)}
    brandVoList: list[brandVo] = [
        brandVo(**{k: v for k, v in row._asdict().items() if k in allowed_keys}) for row in datBrnd.itertuples(index=False)
    ]

    print(f"DataVo Cnt : {len(brandVoList)}")
    # print(brandVoList)

    for batch_num, batch in enumerate(chunks(brandVoList, BATCH_SIZE), 1):
        print(f"▶ Batch {batch_num} / {math.ceil(len(brandVoList) / BATCH_SIZE)}  ({len(batch)} items)")
        # dataclass -> dict
        dict_list = [asdict(vo) for vo in batch]

        # 1. 브랜드 이름 추출 (GPT-o4-mini 입력용)
        brands_names = [vo.brnd_nm for vo in batch]

        # 2. JSON 직렬화 (GPT-4o 입력용)
        brands_json = json.dumps(dict_list, ensure_ascii=False, indent=2)

        print("brands_json : ", brands_json)

        # ## 프롬포트 최초 분석 (o3 모델)
        # messages = [
        #     {"role": "system",
        #      "content": """
        #      너는 대한민국 프랜차이즈 브랜드 정보를 수집·검증하는 리서처다.
        #     반드시 “공식 홈페이지, 공정위 가맹사업 정보공개서, 또는 KOBIS·머니투데이 등
        #     신뢰받는 국내 매체 기사” 1개 이상에서 전화번호를 찾고, 출처 URL 을 함께 기록해라.
        #      """},
        #
        #     {"role": "user", "content": f"""
        #         다음 리스트의 브랜드에 대해 “공식 본사(또는 가맹본부) 영업·대표 ARS” 전화번호를 수집하라.
        #         브랜드명이 동일해도 여러 번호가 있으면 아래 우선순위에 따라 하나만 선택해야 한다.
        #
        #         브랜드 리스트(JSON 배열, brnd_nm 필드가 브랜드명):
        #         {brands_json}
        #
        #         ### 📋 작업 순서 (반드시 지켜라)
        #         1. 브랜드명(brnd_nm)으로 웹 검색해서 알려줘
        #         2. 여럿 발견 시에 중 본사 대표 ARS > 영업 문의 > 고객센터 순으로 대표번호 선택
        #         3. 사용한 URL(페이지 주소)과, 가능하다면 스크린샷 링크(이미지 URL) 함께 기록
        #         4. `지역번호-XXXX-XXXX` 형식(2~3자리 지역번호 + 4자리-4자리)으로 정제
        #         5. JSON 배열 형태로만 출력 (추가 텍스트, 따옴표·``` 블록, 주석 금지)
        #
        #         ### 예시 결과 포맷
        #         [
        #           {{
        #             "brnd_no": "BRD_20140292",
        #             "brnd_nm": "큰맘할매순대국",
        #             "tel_no": "02-1234-5678",
        #             "source": "https://www.eg.co.kr/…"
        #           }},
        #           {{
        #             "brnd_no": "BRD_20171366",
        #             "brnd_nm": "따숩",
        #             "tel_no": "031-5678-1234",
        #             "source": "https://www.eg.co.kr/…"
        #           }},
        #           {{
        #             "brnd_no": "BRD_20220334",
        #             "brnd_nm": "정성카츠",
        #             "tel_no": "010-9876-5432",
        #             "source": "https://www.eg.co.kr/…"
        #           }}
        #         ]
        #
        #         출력 시 ``` 블록, 한글 설명, 주석을 넣지 말고 JSON만 반환해라.
        #     """}
        # ]
        #
        # res = client.chat.completions.create(
        #     model="o4-mini",
        #     messages=messages
        # )



        print("brands_names : ",brands_names)

        ## 프롬포트 최초 분석 (o3 모델)
        messages = [
            {"role": "user", "content": f"""
                {brands_names} 대표전화번호 알려줘
            """}
        ]

        res = client.chat.completions.create(
            model="o3",
            messages=messages
        )

        ## 질의 데이터 가공 (4o 모델)
        resTelNoData = res.choices[0].message.content
        print("1. RES TEL_NO DATA", resTelNoData)

        messages.extend([
            {"role": "assistant", "content": f"""{resTelNoData}"""},
            {"role": "user", "content": 
                f"""
                {brands_json}
                여기 JSON에 방금 너가 찾은 데이터를 매핑해서 반환해
                
                밑에 내가 보여주는 건 너가 출력할 결과 JSON 포맷 예시 데이터야
                [
                  {{
                    "brnd_no": "BRD_20140292",
                    "brnd_nm": "큰맘할매순대국",
                    "tel_no": "02-1234-5678",
                    "source": "https://www.eg.co.kr/…"
                  }},
                  {{
                    "brnd_no": "BRD_20171366",
                    "brnd_nm": "BHC",
                    "tel_no": "031-5678-1234",
                    "source": "https://www.eg.co.kr/…"
                  }},
                  {{
                    "brnd_no": "BRD_20220334",
                    "brnd_nm": "정성카츠",
                    "tel_no": "010-9876-5432",
                    "source": "https://www.eg.co.kr/…"
                  }}
                ]

                반드 시 ``` 블록, 한글 설명, 주석을 넣지 말고 JSON만 반환해
            """}
        ])

        res = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        # 1️⃣ GPT 응답 파싱 (``json 제거 등)
        brandDataByGpt = res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        print("2. GPT JSON DATA:", brandDataByGpt)

        # 2️⃣ JSON 문자열 -> 리스트[dict]
        try:
            batch_dicts = json.loads(brandDataByGpt)
            batch_vo: list[brandVo] = [brandVo(**d) for d in batch_dicts]
            resBrandVoList.extend(batch_vo)
        except json.JSONDecodeError as e:
            print("JSON 파싱 실패:", e)

    # 3️⃣ dataclass 리스트로 변환
    json_out = [asdict(vo) for vo in resBrandVoList]
    with open(DATA_PATH / "brandDataTelNo.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # generateBrand(brandName = "국수나무")
    generateTelNo()