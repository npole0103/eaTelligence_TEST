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

client = OpenAI(api_key=CHATGPT_API_KEY)

@dataclass(frozen=False)
class FranchiseStore:
    franchise_name: str       # 프랜차이즈명
    brand_name: str           # 브랜드명
    store_count: int          # 가맹점수
    business_type: str        # 업종종류
    location: str             # 위치
    monthly_sales: Dict[str, float]  # 1~12월 월매출 (배열 12개)

brandName = "국수나무"

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




