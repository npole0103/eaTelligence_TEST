from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM, formatPercent, \
	formatCountUnit, formatRank, formatRankWithTotal
from src.common.dataConverter import rowToObject
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta, Page4

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
    "uj3_store_cnt_priority": "uj3_store_cnt_priority",
    "uj3_amt_priority": "uj3_amt_priority",
    "store_data": "store_data",
    "sales_data": "sales_data",
    "ai_summary": "ai_summary",
    "kiosk_rate": "kiosk_rate",
    "tablet_rate": "tablet_rate",
    "phone_rate": "phone_rate",
    "etc_rate": "etc_rate"
}

def fomattingPage4(page4: Page4) -> Page4:
    print("================ PAGE2 FORMATTING DATA.. ================")
    print(page4)
    return page4

def page4Main(brnd_no: str, pivotYm: int | str, meta: Meta) -> Page4:
    page4Data: dict = {}
    page4 = rowToObject(page4Data, Page4, column_mapping)

    print("================ PAGE2 RAW DATA.. ================")
    print(page4)
    return page4