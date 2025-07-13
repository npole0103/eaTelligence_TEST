import dataclasses
import os as os
from pathlib import Path
import logging

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 글로벌 변수
ROOT_PATH = Path.cwd()
DATA_PATH = ROOT_PATH / 'dat'
PDF_PATH = ROOT_PATH / 'pdf'
CHUNK_SIZE = 100

# 프롬포트 기본

JSON_RETURN_TEXT = '''
'''

# relevance & hallucination 고려
CHROMA_FRONT_TEXT = '''
     너는 유저로부터 query 를 입력 받아서 최종적인 답변을 해주는 시스템이야.

     그런데, retrieval 된 결과를 본 후에, relevance 가 있는 retrieved chunk 만을 참고해서 최종적인 답변을 해줄거야.
     아래 step 대로 수행한 후에 최종적인 답변을 출력해.

     Step1) retrieved chunk 를 보고 관련이 있으면 각각의 청크 번호 별로 relevance: yes 관련이 없으면 relevance: no 의 형식으로 json 방식으로 출력해.\n
     Step2) 만약 모든 chunk 가 relevance 가 no 인 경우에, relevant chunk 를 찾을 수 없었습니다. 라는 메시지와 함께 프로그램을 종료해.
     Step3) 만약 하나 이상의 chunk 가 relevance 가 yes 인 경우에, 그 chunk들만 참고해서 답변을 생성해.
     Step4) 최종 답변에 hallucination 이 있었는지를 평가하고 만약, 있었다면 hallucination : yes 라고 출력해. 없다면 hallucination : no 라고 출력해.
     Step5) 만약 hallucination : no 인 경우에, Step3로 돌아가서 답변을 다시 생성해. 이 과정은 딱 1번만 실행해 무한루프가 돌지 않도록.
     Step6) 지금까지의 정보를 종합해서 최종적인 답변을 생성해

     답변은 각 스텝 별로 상세하게 출력해

     아래는 user query 와 retrieved chunk 에 대한 정보야.
'''

# 데이터 클래스
@dataclass(frozen=False)
class brandStatsVo:
    uj3_cd: str
    fchhq_no: str
    brnd_no: str
    ym_date: str
    store_open_cnt: int
    store_close_cnt: int
    store_cnt: int
    store_cnt_nice: int
    amt_avg: int
    amt_total: int
    brnd_nm: str
    fchhq_nm: str
    uj3_nm: str

@dataclass(frozen=False)
class datStoreVo:
    pnu: str
    gps_lat: str
    gps_lon: str

    dong_cd: str
    zone_nm: str

    fchhq_no: str
    fchhq_nm: str
    fchhq_rprsv_nm: str

    brnd_no: str
    brnd_nm: str
    brnd_rprsv_nm: str
    ymd_brnd: str

    uj3_cd: str
    uj3_nm: str
    ym_start: str
    ym_end: str

@dataclass(frozen=False)
class datSalesVo:
    uj3_cd: str
    uj3_nm: str
    fchhq_no: str
    fchhq_nm: str
    brnd_no: str
    brnd_nm: str
    ym_sales: str
    dong_cd: str

    zone_nm: str
    zone_cnt: int
    zone_amt_avg: int
    zone_amt_25pct: int
    zone_amt_50pct: int
    zone_amt_75pct: int
    zone_amt_percnt: int

    cty_nm: str
    cty_cnt: int
    cty_amt_25pct: int
    cty_amt_avg: int
    cty_amt_50pct: int
    cty_amt_75pct: int
    cty_amt_percnt: int

    mega_nm: str
    mega_cnt: int
    mega_amt_avg: int
    mega_amt_25pct: int
    mega_amt_50pct: int
    mega_amt_75pct: int
    mega_amt_percnt: int

    all_cnt: int
    all_amt_avg: int
    all_amt_25pct: int
    all_amt_50pct: int
    all_amt_75pct: int
    all_amt_percnt: int