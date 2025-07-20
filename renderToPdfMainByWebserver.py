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

def run_static_server(directory: Path, port=8000):
    """HTTPServer 객체와 스레드를 반환"""
    import os
    os.chdir(directory)
    server = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(1)  # 서버 준비 대기
    return server, thread

def render_with_js_and_export_pdf(brnd_no: str):

    # 간이 웹 서버 시작
    server, thread = run_static_server(RESOURCE_PATH)

    OUTPUT_FILE_NAME = "index.pdf"
    output_path = OUTPUT_PATH / OUTPUT_FILE_NAME

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # 콘솔 로그 확인
        page.on("console", lambda msg: print(f"[console] {msg.type}: {msg.text}"))
        page.on("requestfailed", lambda req: print(f"[404] {req.url}"))

        # HTML 열기 (fetch가 작동하려면 http://로 열어야 함)
        page.goto(f"http://localhost:8000/resources/index.html")

        # JSON이 렌더링될 때까지 대기
        page.wait_for_timeout(5000)

        # PDF 저장
        page.pdf(path=str(output_path), format="A4", print_background=True)
        browser.close()

    # ✅ 서버 종료
    server.shutdown()
    thread.join()

    print(f"✅ PDF 저장 완료: {output_path}")

if __name__ == "__main__":
    render_with_js_and_export_pdf("bhc")