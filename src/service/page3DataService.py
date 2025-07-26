from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM, formatPercent, \
	formatCountUnit, formatRank, formatRankWithTotal
from src.common.dataConverter import rowToObject
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta, Page3, SalesTrend, RankByCity
import pandas as pd
from typing import List

from datetime import datetime

from src.service.metaDataService import metaMain

# datFchhq = load_excel("datFchhq")
# datBrnd = load_excel("datBrnd")
# datStore = load_excel("datStore")
# datStore = load_excel("datStore")
# datSales = load_excel("datSales")
# datKeyword = load_excel("datKeyword")

brandStats = load_excel("brandStats")
datSalesAppend = load_excel("datSalesAppend")
cdDong = load_excel("cdDong")

# LEFT : PANDAS / RIGHT : DATACLASS
column_mapping = {
    "ai_summary": "ai_summary",
    "m_avg_amt": "m_avg_amt",
    "y_avg_amt_per_store": "y_avg_amt_per_store",
    "m_avg_amt_per_store": "m_avg_amt_per_store",
    "amt_rank_by_city": "amt_rank_by_city",
    "sales_trend": "sales_trend"
}

def add_months(year: int, month: int, i: int) -> tuple[int, int]:
    """
    연도와 월을 기준으로 i개월 후의 (연, 월) 반환
    """
    total_month = month + i
    new_year = year + (total_month - 1) // 12
    new_month = (total_month - 1) % 12 + 1
    return new_year, new_month

def getMonthlyAvgAmt(brnd_no: str, pivotYm: int) -> int:
    # pivotYm에서 연도 추출
    year = pivotYm // 100

    # 해당 연도의 1월~12월 범위 계산
    startYm = year * 100 + 1  # 예: 202401
    endYm = year * 100 + 12  # 예: 202412

    # 필터링
    df_year = brandStats[
        (brandStats['brnd_no'] == brnd_no) &
        (brandStats['ym_sales'] >= startYm) &
        (brandStats['ym_sales'] <= endYm)
        ].copy()

    # ③ amt_avg * store_cnt_nice 총합 계산
    df_year['weighted_amt'] = df_year['amt_avg'] * df_year['store_cnt_nice']
    m_avg_amt = df_year['weighted_amt'].mean()

    return int(m_avg_amt)

def getYearlyAvgAmtPerStore(brnd_no: str, pivotYm: int) -> int:
    # pivotYm에서 연도 추출
    year = pivotYm // 100

    # 해당 연도의 1월~12월 범위 계산
    startYm = year * 100 + 1  # 예: 202401
    endYm = year * 100 + 12  # 예: 202412

    # 필터링
    df_year = brandStats[
        (brandStats['brnd_no'] == brnd_no) &
        (brandStats['ym_sales'] >= startYm) &
        (brandStats['ym_sales'] <= endYm)
        ].copy()

    y_avg_amt_per_store = df_year['amt_avg'].sum()

    return int(y_avg_amt_per_store)

def getMonthlyAvgAmtPerStore(brnd_no: str, pivotYm: int) -> int:
    # pivotYm에서 연도 추출
    year = pivotYm // 100

    # 해당 연도의 1월~12월 범위 계산
    startYm = year * 100 + 1  # 예: 202401
    endYm = year * 100 + 12  # 예: 202412

    # 필터링
    df_year = brandStats[
        (brandStats['brnd_no'] == brnd_no) &
        (brandStats['ym_sales'] >= startYm) &
        (brandStats['ym_sales'] <= endYm)
        ].copy()

    m_avg_amt_per_store = df_year['amt_avg'].mean()

    return int(m_avg_amt_per_store)

def getSalesTrend(brnd_no: str, pivotYm: int | str) -> SalesTrend:
    pivot_year = pivotYm // 100

    # ① 24.01 ~ 24.12 → 12개월 + 예상 한 달
    start_year, start_month = pivot_year, 1
    months = 13

    ym_list = []
    for i in range(months):
        y, m = add_months(start_year, start_month, i)
        ym_list.append(y * 100 + m)

    # ② 해당 데이터만 필터링
    df_filtered = brandStats[
        (brandStats['brnd_no'] == brnd_no) &
        (brandStats['ym_sales'].isin(ym_list))
    ].copy()

    # ③ 매출 계산: amt_avg * store_cnt_nice
    sales_by_ym = {
        row['ym_sales']: row['amt_avg'] * row['store_cnt_nice']
        for _, row in df_filtered.iterrows()
    }

    # ④ x/y 축 생성
    xAxisData = []
    yAxisData = []

    for i, ym in enumerate(ym_list):
        if i < 12:
            label = f"{str(ym)[2:4]}.{str(ym)[4:]:0>2}"  # 예: 24.01
        else:
            label = "예상"  # 13번째 항목

        xAxisData.append(label)
        yAxisData.append(int(sales_by_ym.get(ym, 0)))

    return SalesTrend(xAxisData=xAxisData, yAxisData=yAxisData)

def getRankAmtByCity(brnd_no: str, pivotYm: int | str) -> List[RankByCity]:
    filtered = datSalesAppend[
        (datSalesAppend['brnd_no'] == brnd_no) &
        (datSalesAppend['ym_sales'] == pivotYm)
    ].copy()

    if filtered.empty:
        return [
            RankByCity(rank='-', region='-', store_count='-') for _ in range(10)
        ]

    # 2. 금액 계산
    filtered['zone_amt_total'] = (filtered['zone_cnt'] * filtered['zone_amt_avg']).astype(int)

    # 3. 매핑: dong_cd → cty_nm
    dong_to_city = dict(zip(cdDong['dong_cd'], cdDong['cty_nm']))
    filtered['cty_nm'] = filtered['dong_cd'].map(dong_to_city)

    # 4. 지역별 금액 합산
    city_totals = filtered.groupby('cty_nm')['zone_amt_total'].sum().reset_index()
    city_totals.columns = ['region', 'store_count']  # store_count 자리에 금액을 둠

    # 5. 공동 순위 부여
    city_totals['rank'] = city_totals['store_count'].rank(method='min', ascending=False).astype(int)

    # 6. 정렬
    city_totals.sort_values(['rank', 'region'], inplace=True)

    # 7. 상위 10개만 자르기
    top10_df = city_totals.head(10)

    # 8. 부족한 경우 '-'로 채우기
    while len(top10_df) < 10:
        top10_df = pd.concat([
            top10_df,
            pd.DataFrame([{'rank': '-', 'region': '-', 'store_count': '-'}])
        ], ignore_index=True)

    # 9. dataclass로 변환
    amt_rank_by_city = [
        RankByCity(rank=row['rank'], region=row['region'], store_count=row['store_count'])
        for _, row in top10_df.iterrows()
    ]

    return amt_rank_by_city

def fomattingPage3(page3: Page3) -> Page3:
    print("================ PAGE2 FORMATTING DATA.. ================")

    page3.m_avg_amt = formatKoreanCurrency(page3.m_avg_amt)

    print(page3)
    return page3

def page3Main(brnd_no: str, pivotYm: int | str, meta: Meta) -> Page3:
    page3Data: dict = {}
    page3 = rowToObject(page3Data, Page3, column_mapping)
    page3.m_avg_amt = getMonthlyAvgAmt(brnd_no, pivotYm)
    page3.y_avg_amt_per_store = getYearlyAvgAmtPerStore(brnd_no, pivotYm)
    page3.m_avg_amt_per_store = getMonthlyAvgAmtPerStore(brnd_no, pivotYm)
    page3.sales_trend = getSalesTrend(brnd_no, pivotYm)
    page3.amt_rank_by_city = getRankAmtByCity(brnd_no, pivotYm)

    print("================ PAGE2 RAW DATA.. ================")
    print(page3)
    return page3

if __name__ == "__main__":
    meta = metaMain("BRD_20190835", 202412)
    page3Main('BRD_20190835', 202412, meta)