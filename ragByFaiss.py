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

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def	datToText(df):
    text = f'''
    {df['aa']}
    ''' # TODO 칼럼 매핑
    return text

# 파일 불러오기
print("파일 불러오기 텍스트..")
datSales = ""
df = pd.DataFrame(datSales)

# 데이터 파일 TEXT화 가공
print("DAT 파일 데이터 텍스트..")
df["text"] = df.apply(
    lambda row: f"{row['region']} 지역, {row['stores']}개 가맹점, 월 매출 {row['sales']}원.",
    axis=1 # 행 방향(axis) == 1
)

# 임베딩 작업 진행
print("임베딩 작업 진행..")
df["embedding"] = df["text"].apply(lambda x: get_embedding(x))

# 데이터 CSV 파일 저장
print("데이터 CSV 파일 저장..")
df.to_csv("embedding_data.csv", index=False, encoding="utf-8")




