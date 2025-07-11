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

# 1. 여러 PDF 로드
def load_pdfs(pdf_dir):
    docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            docs.extend(loader.load())
    return docs

# 2. 텍스트 분할 (RecursiveCharacterTextSplitter 사용)
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# 3. 전체 처리 파이프라인
def build_chroma_vectorstore(pdf_dir, persist_dir):
    # (1) PDF 로딩
    documents = load_pdfs(pdf_dir)

    # (2) 텍스트 분할
    split_docs = split_documents(documents)

    # (3) OpenAI 임베딩
    embedding = OpenAIEmbeddings(openai_api_key=CHATGPT_API_KEY)

    # (4) Chroma에 저장
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"✅ {len(split_docs)} chunks saved to Chroma!")

# 실행
build_chroma_vectorstore(pdf_dir="./pdf", persist_dir="./chroma_db")