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

## DAT_BRND 가공
# datUj = pd.read_csv(DATA_PATH / "cd_uj.dat", sep='|', encoding='utf-8')
# datBrnd = pd.read_csv(DATA_PATH / "dat_brnd.dat", sep='|', encoding='utf-8')
# 
# datBrnd = datBrnd.merge(datUj, on='uj3_cd', how='left')
# 
# datBrnd["tel_no"] = ""
# 
# datBrnd["store_cnt"] = datBrnd["store_cnt"].apply(
#     lambda x: f"{int(x)}" if pd.notna(x) else "")
# 
# datBrnd["ymd_brnd"] = datBrnd["ymd_brnd"].apply(
#     lambda x: f"{int(x)}" if pd.notna(x) else "")
# 
# datBrnd["is_submit"] = datBrnd["is_submit"].apply(
#     lambda x: f"{int(x)}" if pd.notna(x) else "")
# 
# (datBrnd[['fchhq_no', 'brnd_no', 'y_ftc', 'brnd_nm', 'rprsv_nm', 'ymd_brnd', 'store_cnt', 'uj3_cd', 'uj3_nm', 'uj2_cd', 'uj2_nm', 'tel_no', 'is_submit']]
#  .to_csv(DATA_PATH / "dat_brnd_v2.dat", sep="|", index=False, encoding="utf-8"))

## dat_fchhq 가공
dat_fchhq = pd.read_csv(DATA_PATH / "dat_fchhq.dat", sep='|', encoding='utf-8')

(dat_fchhq[['fchhq_no', 'y_ftc', 'fchhq_nm', 'rprsv_nm']]
 .to_csv(DATA_PATH / "dat_fchhq_v2.dat", sep="|", index=False, encoding="utf-8"))

