from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM, formatPercent, \
	formatCountUnit, formatRank, formatRankWithTotal
from src.common.dataConverter import rowToObject
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta, Page3

from datetime import datetime

from src.service.metaDataService import metaMain

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
    "m_avg_amt": "m_avg_amt",
    "y_avg_amt_per_store": "y_avg_amt_per_store",
    "m_avg_amt_per_store": "m_avg_amt_per_store",
    "amt_rank_by_city": "amt_rank_by_city",
    "sales_trend": "sales_trend"
}

def fomattingPage3(page3: Page3) -> Page3:
    print("================ PAGE2 FORMATTING DATA.. ================")
    print(page3)
    return page3

def page3Main(brnd_no: str, pivotYm: int | str, meta: Meta) -> Page3:
    page3Data: dict = {}
    page3 = rowToObject(page3Data, Page3, column_mapping)

    print("================ PAGE2 RAW DATA.. ================")
    print(page3)
    return page3

if __name__ == "__main__":
    meta = metaMain("BRD_20190835", 202412)
    page3Main('BRD_20190835', 202412, meta)