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

def renderingLogo(brnd_no :str):
    LOGO_FILE_NAME = brnd_no + ".png"

    # HTML 불러오기
    with open(RESOURCE_PATH / "template.html", "r", encoding="utf-8") as f:
        html_template = Template(f.read())

    # 동적으로 경로 주입
    rendered_html = html_template.substitute(
        image_path = Path("../logo") / LOGO_FILE_NAME
    )

    # 결과 HTML 저장
    with open(RESOURCE_PATH / f"{brnd_no}.html", "w", encoding="utf-8") as f:
        f.write(rendered_html)

def render_with_js_and_export_pdf(brnd_no: str):
    with sync_playwright() as p:
        OUTPUT_FILE_NAME = brnd_no + ".pdf"

        # 로고 데이터 매핑
        renderingLogo(brnd_no)

        browser = p.chromium.launch()
        page = browser.new_page()

        # JS 오류 콘솔 확인
        page.on("console", lambda msg: print(f"[console] {msg.type}: {msg.text}"))

        # file:// 경로로 로드 (JS 동작 포함, 이미지도 로컬 가능)
        page.goto((RESOURCE_PATH / f"{brnd_no}.html").as_uri())

        # JavaScript가 JSON을 렌더링할 시간 기다림
        page.wait_for_timeout(5000)  # 혹은 page.wait_for_selector("#content > div") 등 사용 가능
        # page.wait_for_selector("#content > .card")

        # PDF로 저장
        page.pdf(
            path=str(OUTPUT_PATH / OUTPUT_FILE_NAME),
            format="A4",
            print_background=True  # 이미지와 배경 포함
        )
        browser.close()
        return OUTPUT_FILE_NAME

if __name__ == "__main__":
    outputFilename = render_with_js_and_export_pdf("bhc")
    print(f"✅ PDF 저장 완료: {OUTPUT_PATH / outputFilename}")