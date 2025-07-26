from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM, formatPercent, \
	formatCountUnit, formatRank, formatRankWithTotal
from src.common.dataConverter import rowToObject
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta, Page5

from datetime import datetime

# datFchhq = load_excel("datFchhq")
# datBrnd = load_excel("datBrnd")
# datStore = load_excel("datStore")
# brandStats = load_excel("brandStats")
# datStore = load_excel("datStore")
# datSales = load_excel("datSales")
# datKeyword = load_excel("datKeyword")

# LEFT : PANDAS / RIGHT : DATACLASS
column_mapping = {
    "ai_summary": "ai_summary",
    "used": "used",
    "unused": "unused",
    "keyword_search_cnt": "keyword_search_cnt",
    "blog_post_cnt": "blog_post_cnt",
    "cafe_post_cnt": "cafe_post_cnt",
    "ai_news": "ai_news"
}

def fomattingPage5(page5: Page5) -> Page5:
    print("================ PAGE2 FORMATTING DATA.. ================")
    print(page5)
    return page5

def page5Main(brnd_no: str, pivotYm: int | str, meta: Meta) -> Page5:
    page5Data: dict = {}
    page5 = rowToObject(page5Data, Page5, column_mapping)

    print("================ PAGE2 RAW DATA.. ================")
    print(page5)
    return page5