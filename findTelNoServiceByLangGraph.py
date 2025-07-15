import dataclasses
import logging
import os
from pathlib import Path
import re
import gc
import math

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

from common.module import DATA_PATH, brandStatsVo, datStoreVo, datSalesVo

import warnings

warnings.filterwarnings(
	"ignore",
	category=DeprecationWarning,
	module=r"langchain_core\.tools",
)

load_dotenv()
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# ─── 1. 데이터 클래스 ──────────────────────────────
@dataclass
class BrandVo:
	brnd_no: str
	brnd_nm: str
	tel_no: Optional[str] = None
	source: Optional[str] = None


# ─── 2. LLM ───────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=CHATGPT_API_KEY)

# ─── 3. 검색‧브라우저 Tool 준비 ───────────────────
# search_tool = DuckDuckGoSearchRun()                     # ⓐ 검색 전용
# async_browser = create_async_playwright_browser()
# sync_browser  = create_sync_playwright_browser()
#
# pw_tools = PlayWrightBrowserToolkit.from_browser(       # ⓑ Playwright 툴 모음
#     async_browser=async_browser,
#     sync_browser=sync_browser,
# ).get_tools()
#
# # ZeroShot 제한을 피하기 위해 Functions Agent 사용 → multi-input 허용
# tools_for_agent = [search_tool] + [
#     t for t in pw_tools if t.name != "previous_webpage"  # prev_webpage 제거
# ]

# search_tool = GoogleSerperRun()
google_search = GoogleSerperAPIWrapper()
tools = [Tool(
	name="IntermediateAnswer",
	func=google_search.run,
	description="useful for when you need to ask with search")
]

agent = initialize_agent(
#	tools=tools_for_agent,
	tools=tools,
	llm=llm,
	agent=AgentType.OPENAI_FUNCTIONS,  # Functions 기반
	verbose=True,
)


# ─── 4. find_brand_tel Tool (Agent 래핑) ───────────
def _find_tel(brand_name: str) -> dict:
	prompt = (
		f"""
		너는 프랜차이즈 분석 봇이다. 다음 브랜드의 본사(가맹본부) 대표 전화번호를 찾아라.
		
		검색은 '[브랜드명] 음식브랜드 가맹 대표전화번호' 양식으로 검색해 줘
		① 공식 홈페이지 ② 공정위 가맹사업정보공개서 ③ 언론 기사 중 하나 이상 열람해
		URL(https://…)과 전화번호(지역번호-XXXX-XXXX)만 순수 텍스트로 반환.
		필요 시 "정보공개서 PDF" 또는 사업자등록번호를 조합해서 대표번호 찾아줘
		
		브랜드명: {brand_name}
		"""
	)

	# Functions Agent는 {"output": "..."} 형태로 답을 돌려준다
	resp_dict = agent.invoke({"input": prompt})
	text = resp_dict.get("output", "")  # ← 문자열만 추출

	url, tel = None, None
	for tok in text.split():
		if tok.startswith(("http://", "https://")):
			url = tok
		if "-" in tok and tok.replace("-", "").isdigit():
			tel = tok

	return {"brand": brand_name, "tel": tel, "source": url}


find_tel_tool = Tool(
	name="find_brand_tel",
	func=_find_tel,
	description="브랜드명을 넣으면 {'brand','tel','source'} 반환",
	is_single_input=True,
)


# ─── 5. LangGraph 정의 ────────────────────────────
@dataclass
class TelState:
	batch: List[BrandVo]
	result: Optional[List[BrandVo]] = None


def search_node(state: TelState) -> TelState:
	out: List[BrandVo] = []
	for vo in state.batch:
		res = find_tel_tool.invoke(vo.brnd_nm)  # .run → .invoke
		out.append(
			BrandVo(
				brnd_no=vo.brnd_no,
				brnd_nm=vo.brnd_nm,
				tel_no=res["tel"],
				source=res["source"],
			)
		)
	return TelState(batch=state.batch, result=out)


graph = StateGraph(TelState)
graph.add_node("search", search_node)
graph.set_entry_point("search")
graph.set_finish_point("search")  # 단일 노드이므로 finish=search
tel_workflow = graph.compile()


# ─── 6. 메인: CSV → 전화번호 JSON ──────────────────
def generate_tel():
	df = pd.read_csv(DATA_PATH / "dat_brnd_v2.csv", encoding="utf-8")
	df = df.query("is_submit == 1")[["brnd_no", "brnd_nm"]]
	vo_list = [BrandVo(r.brnd_no, r.brnd_nm) for r in df.itertuples(index=False)]

	result_state = tel_workflow.invoke(TelState(batch=vo_list))

	all_results = result_state["result"]

	with open(DATA_PATH / "brandDataTelNo.json", "w", encoding="utf-8") as f:
		json.dump([asdict(v) for v in all_results], f, ensure_ascii=False, indent=2)

	print(f"✅ 완료: {len(all_results)}개 브랜드 전화번호 저장")

if __name__ == "__main__":
	generate_tel()
