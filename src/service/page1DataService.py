from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM, formatPercent, \
	formatCountUnit, formatRank, formatRankWithTotal
from src.common.dataConverter import rowToObject
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta, Page1

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
    "brandData": "brandData",
    "averageData": "averageData"
}

def fomattingPage1(page1: Page1) -> Page1:
    print("================ PAGE2 FORMATTING DATA.. ================")
    print(page1)
    return page1

def page1Main(brnd_no: str, pivotYm: int | str, meta: Meta) -> Page1:
    page1Data: dict = {}
    page1 = rowToObject(page1Data, Page1, column_mapping)

    print("================ PAGE2 RAW DATA.. ================")
    print(page1)
    return page1