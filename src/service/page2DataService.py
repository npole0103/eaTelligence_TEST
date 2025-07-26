from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM, formatPercent, \
    formatCountUnit, formatRank, formatRankWithTotal
from src.common.dataConverter import rowToObject
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta, Page2, RegionData, StoreTrend, RankByCity
import pandas as pd
from typing import List

from datetime import datetime

from src.service.metaDataService import metaMain

datStore = load_excel("datStore")
cdDong = load_excel("cdDong")
datStatsDongMapping = load_excel("datStatsDongMapping")
brandStats = load_excel("brandStats")

# LEFT : PANDAS / RIGHT : DATACLASS
column_mapping = {
    "ai_summary": "ai_summary",
    "y_diff_store_cnt": "y_diff_store_cnt",
    "regionData": "regionData",
    "store_trend": "store_trend",
    "store_cnt_rank_by_city": "store_cnt_rank_by_city"
}

def getYoyDiffStoreCnt(meta: Meta) -> int:
    return int(meta.y_open_store - meta.y_close_store)

def countStoreByRegion(brnd_no: str, pivotYm: int | str) -> list[RegionData]:
    # 1. 폐업하지 않은 점포 필터링
    filtered = datStore[
        (datStore['brnd_no'] == brnd_no) &
        (datStore['ym_end'] > pivotYm)
    ]

    # 2. dong_cd -> geo_cd 매핑을 위한 dict 생성
    dong_to_geo = dict(zip(datStatsDongMapping['dong_cd'], datStatsDongMapping['geo_cd']))

    # 3. dong_cd → geo_cd 수동 매핑 (map은 벡터연산, merge보다 빠름)
    filtered = filtered.copy()
    filtered['geo_cd'] = filtered['dong_cd'].map(dong_to_geo)

    # 4. geo_cd 기준 카운트
    geo_counts = filtered['geo_cd'].value_counts()

    # 5. 리스트[RegionData] 형태로 반환
    regionData = [RegionData(name=str(geo_cd), value=int(count)) for geo_cd, count in geo_counts.items() if pd.notna(geo_cd)]
    return regionData


def getStoreTrend(brnd_no: str, pivotYm: int | str) -> StoreTrend:
    base_year = pivotYm // 100  # 예: 2026
    # 1. 1~12월 ym_sale 생성: 202601 ~ 202612
    months = [base_year * 100 + m for m in range(1, 13)]  # [202601, ..., 202612]

    # 2. 데이터 필터링
    filtered = brandStats[
        (brandStats['brnd_no'] == brnd_no) &
        (brandStats['ym_sales'].isin(months))
        ][['ym_sales', 'store_cnt']].copy()

    # 3. index를 ym_sale로 설정
    filtered.set_index('ym_sales', inplace=True)

    # 4. 1~12월 고정된 인덱스로 재배열, 결측치는 0으로
    full_index = pd.Index(months, name='ym_sales')
    aligned = filtered.reindex(full_index).fillna(0)

    # 5. x축: "26.01" ~ "26.12"
    xAxisData = [f"{str(base_year)[2:]}.{str(m)[-2:]}" for m in months]
    yAxisData = aligned['store_cnt'].astype(int).tolist()

    return StoreTrend(xAxisData=xAxisData, yAxisData=yAxisData)

def getRankStoreCountByCity(brnd_no: str) -> List[RankByCity]:
    # 1. brnd_no 필터링
    filtered = datStore[datStore['brnd_no'] == brnd_no].copy()

    if filtered.empty:
        return [
            RankByCity(rank='-', region='-', store_count='-') for _ in range(10)
        ]

    # 2. dong_cd → cty_nm 매핑 dict 생성
    dong_to_city = dict(zip(cdDong['dong_cd'], cdDong['cty_nm']))

    # 3. 매핑 적용
    filtered['cty_nm'] = filtered['dong_cd'].map(dong_to_city)

    # 4. cty_nm별 카운트
    city_counts = filtered['cty_nm'].value_counts().reset_index()
    city_counts.columns = ['region', 'store_count']

    # 5. 공동 순위 부여
    city_counts['rank'] = city_counts['store_count'].rank(method='min', ascending=False).astype(int)

    # 6. 정렬
    city_counts.sort_values(['rank', 'region'], inplace=True)

    # 7. 상위 10개만 자르기
    top10_df = city_counts.head(10)

    # 8. 부족한 경우 나머지 항목 채우기
    while len(top10_df) < 10:
        top10_df = pd.concat([
            top10_df,
            pd.DataFrame([{'rank': '-', 'region': '-', 'store_count': '-'}])
        ], ignore_index=True)

    # 9. dataclass 변환
    store_cnt_rank_by_city = [
        RankByCity(rank=row['rank'], region=row['region'], store_count=row['store_count'])
        for _, row in top10_df.iterrows()
    ]
    return store_cnt_rank_by_city


def fomattingPage2(page2: Page2) -> Page2:
    print("================ PAGE2 FORMATTING DATA.. ================")
    print(page2)
    return page2

def page2Main(brnd_no: str, pivotYm: int | str, meta: Meta) -> Page2:
    page2Data: dict = {}
    page2 = rowToObject(page2Data, Page2, column_mapping)
    page2.y_diff_store_cnt = getYoyDiffStoreCnt(meta)
    page2.regionData = countStoreByRegion(brnd_no, pivotYm)
    page2.store_trend = getStoreTrend(brnd_no, pivotYm)
    page2.store_cnt_rank_by_city = getRankStoreCountByCity(brnd_no)

    print("================ PAGE2 RAW DATA.. ================")
    print(page2)
    return page2

if __name__ == "__main__":
    meta = metaMain("BRD_20190835", 202412)
    page2Main('BRD_20190835', 202412, meta)