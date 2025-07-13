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

# Chroma DB 불러오기
def load_chroma(persist_dir="./chroma_db"):
    embedding = OpenAIEmbeddings(openai_api_key=CHATGPT_API_KEY)
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    return db

# 질의 실행
def query_documents(db, query, k=3):
    # MMR(Max Marginal Relevance)
    '''
        MMR : 중복 문서를 피하면서 다양하고 관련성 높은 문서 유지
        k : 유사한 문서 반환 갯수
        fetch_k : 후보로 가져오는 유사한 문서 갯수
        score_threshold : 유사도 점수 기준 필터링 (0~1)
    '''
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 25, "score_threshold": 0.8})
    results = retriever.get_relevant_documents(query)

    for i, doc in enumerate(results, 1):
        print(f"\n🔎 결과 {i}:")
        print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
        print(doc.page_content[:500] + "...")
        print("-" * 50)

if __name__ == "__main__":
    db = load_chroma(persist_dir="./chroma_db")
    user_query = input("질문을 입력하세요: ")
    query_documents(db, user_query)