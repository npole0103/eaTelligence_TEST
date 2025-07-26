from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM, formatPercent, \
	formatCountUnit, formatRank, formatRankWithTotal
from src.common.dataConverter import rowToObject
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta

from datetime import datetime

datFchhq = load_excel("datFchhq")
datBrnd = load_excel("datBrnd")
datStore = load_excel("datStore")
brandStats = load_excel("brandStats")
datStore = load_excel("datStore")
datSales = load_excel("datSales")
datKeyword = load_excel("datKeyword")
datSalesAppend = load_excel("datSalesAppend")

# LEFT : PANDAS / RIGHT : DATACLASS
column_mapping = {
	"brnd_no": "brnd_no",
	"brnd_nm": "brnd_nm",
	"uj3_nm": "uj3_nm",
	"fchhq_nm": "fchhq_nm",
	"rprsv_nm": "brnd_rprsv_nm",
	"tel_no": "tel_no",
	"brno": "brnd_brno",
	"ymd_brnd": "ymd_brnd",
	"opr_duration": "opr_duration",
	"brnd_store_cnt": "brnd_store_cnt",
	"y_total_amt": "y_total_amt",

	"market_uj3_total": "market_uj3_total",
	"market_amt_rate": "market_amt_rate",
	"market_store_cnt_rate": "market_store_cnt_rate",
	"market_amt_rank": "market_amt_rank",
	"market_store_cnt_rank": "market_store_cnt_rank",

	"y_store_survival": "y_store_survival",
	"y_amg_inc_dec_rate": "y_amg_inc_dec_rate",
	"y_store_cnt_inc_dec_rate": "y_store_cnt_inc_dec_rate",

	"amt_top1_store_rate": "amt_top1_store_rate",
	"amt_top2_store_rate": "amt_top2_store_rate",
	"amt_top3_store_rate": "amt_top3_store_rate",
	"amt_top4_store_rate": "amt_top4_store_rate",

	"y_portal_search_cnt": "y_portal_search_cnt",

	"y_open_store": "y_open_store",
	"y_close_store": "y_close_store"
}

def addDictToDataClass(dataClass, dict: dict):
    for key, value in dict.items():
        setattr(dataClass, key, value)
    return dataClass

def _safe_cast(value):
    # None은 그대로
    if value is None:
        return None
    # float이지만 정수로 표현 가능하면 int로
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

def getMetaDefaultData(brnd_no: str) -> Meta:
	datBrnd_filter = datBrnd[datBrnd['brnd_no'] == brnd_no]
	meta_instance = rowToObject(datBrnd_filter.iloc[0].to_dict(), Meta, column_mapping)

	return meta_instance

def getFchhqName(brnd_no: str) -> str:
	datBrnd_filter = datBrnd[datBrnd['brnd_no'] == brnd_no]
	fchhq_no_by_datBrand = datBrnd_filter.iloc[0]['fchhq_no']
	datFchhq_filter = datFchhq[datFchhq['fchhq_no'] == fchhq_no_by_datBrand]

	return datFchhq_filter.iloc[0]['fchhq_nm']

def getElapsedYearsMonths(ymd_brnd: int) -> int:
    date_str = str(ymd_brnd)
    start_date = datetime.strptime(date_str, "%Y%m%d")

    today = datetime.today()

    years = today.year - start_date.year
    months = today.month - start_date.month

    if months < 0:
        years -= 1
        months += 12

    return years * 12 + months

def getAliveStoreCount(brndNo: str, pivotYm: int) -> int:
    aliveStores = datStore[
        (datStore['brnd_no'] == brndNo) &
        (datStore['ym_end'] > pivotYm)
    ]
    return aliveStores.shape[0]

def getYoYTotalAmt(brnd_no: str, pivotYm: str | int) -> int:
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
	total_amt = df_year['weighted_amt'].sum()

	return int(total_amt)

def marketDataByBranStats(brnd_no: str, pivotYm: str | int):
	# 1. 대상 브랜드의 업종 코드 찾기
	target_row = brandStats[brandStats['brnd_no'] == brnd_no]
	if target_row.empty:
		raise ValueError("해당 브랜드가 존재하지 않습니다.")

	target_uj3_cd = target_row.iloc[0]['uj3_cd']

	# 2. 해당 업종, 해당 월 필터링
	filtered = brandStats[(brandStats['uj3_cd'] == target_uj3_cd) & (brandStats['ym_sales'] == pivotYm)].copy()

	# 3. 전체 합계
	total_amt_avg = filtered['amt_avg'].sum()
	total_store_cnt = filtered['store_cnt'].sum()

	# 4. 대상 브랜드 데이터
	target_data = filtered[filtered['brnd_no'] == brnd_no]
	if target_data.empty:
		raise ValueError(f"해당 브랜드는 {pivotYm}월에 데이터가 없습니다.")

	# 5. 퍼센트 계산
	amt_avg_pct = round(target_data.iloc[0]['amt_avg'] / total_amt_avg * 100, 2)
	store_cnt_pct = round(target_data.iloc[0]['store_cnt'] / total_store_cnt * 100, 2)

	# 6. 등수 계산 (동순위 인정)
	filtered['amt_avg_rank'] = filtered['amt_avg'].rank(method='min', ascending=False)
	filtered['store_cnt_rank'] = filtered['store_cnt'].rank(method='min', ascending=False)

	total_count = len(filtered)

	target_amt_rank = int(filtered[filtered['brnd_no'] == brnd_no]['amt_avg_rank'].iloc[0])
	target_store_rank = int(filtered[filtered['brnd_no'] == brnd_no]['store_cnt_rank'].iloc[0])

	# 7. 결과 반환
	return {
		'market_uj3_total': total_count,
		'market_amt_rate': float(amt_avg_pct),
		'market_store_cnt_rate': float(amt_avg_pct),
		'market_amt_rank': target_amt_rank,
		'market_store_cnt_rank': target_store_rank
	}

def getAvgStoreLife(brnd_no: str) -> int:
	# 1. 브랜드 필터링
	df = datStore[datStore['brnd_no'] == brnd_no]

	# 2. ym 필드가 결측치가 아닌 것만 필터링
	df = df[df['ym_start'].notna() & df['ym_end'].notna()]

	# 3. ym_end - ym_start 계산 (YYYYMM 기준 → 개월로 변환)
	def ym_diff(row):
		start = int(row['ym_start'])
		end = int(row['ym_end'])
		return (end // 100 - start // 100) * 12 + (end % 100 - start % 100)

	df['duration_months'] = df.apply(ym_diff, axis=1)

	# 4. 평균 수명 (개월)
	return int(df['duration_months'].mean())

def getYoyChangeRate(brnd_no: str, pivotYm: int | str) -> dict:
    prevPivotYm = pivotYm - 100  # 1년 전 연월 (예: 202412 → 202312)

    # 1. 필터링
    this_year = brandStats[(brandStats['brnd_no'] == brnd_no) & (brandStats['ym_sales'] == pivotYm)]
    last_year = brandStats[(brandStats['brnd_no'] == brnd_no) & (brandStats['ym_sales'] == prevPivotYm)]

    if this_year.empty or last_year.empty:
        return {
            'y_amg_inc_dec_rate': None,
            'y_store_cnt_inc_dec_rate': None
        }

    # 2. 값 추출
    amt_avg_now = this_year.iloc[0]['amt_avg']
    amt_avg_prev = last_year.iloc[0]['amt_avg']
    store_cnt_now = this_year.iloc[0]['store_cnt']
    store_cnt_prev = last_year.iloc[0]['store_cnt']

    # 3. 증감률 계산 함수
    def calc_rate(now, prev) -> float:
        if prev == 0:
            return None
        return round((now - prev) / prev * 100, 1)  # float 퍼센트 반환

    return {
        'y_amg_inc_dec_rate': float(calc_rate(amt_avg_now, amt_avg_prev)),
        'y_store_cnt_inc_dec_rate': float(calc_rate(store_cnt_now, store_cnt_prev))
    }


def getAmtTopStoreRates(brnd_no: str, pivotYm: int) -> dict:
	# 1. 해당 브랜드의 업종 코드 찾기
	row = brandStats[brandStats['brnd_no'] == brnd_no]
	if row.empty:
		return {"error": "브랜드를 찾을 수 없음"}

	uj3_cd = row.iloc[0]['uj3_cd']

	# 2. 업종 전체 기준분포값 찾기
	dist_row = datSales[(datSales['uj3_cd'] == uj3_cd)
						& (datSales['ym_sales'] == pivotYm)
						& (datSales['fchhq_no'].isna())
						& (datSales['brnd_no'].isna())]
	if dist_row.empty:
		print("datSales null")
		return {"error": "해당 업종의 매출 분위 정보가 없음"}

	pct_25 = dist_row.iloc[0]['all_amt_25pct']
	pct_50 = dist_row.iloc[0]['all_amt_50pct']
	pct_75 = dist_row.iloc[0]['all_amt_75pct']

	# # 3. brandStats에서 업종 + 월 데이터만 필터
	# brand_filtered = brandStats[(brandStats['uj3_cd'] == uj3_cd) &
	# 							(brandStats['ym_sales'] == pivotYm)]
	# if brand_filtered.empty:
	# 	print("datStats null")
	# 	return {"error": "브랜드의 해당 월 점포 데이터가 없음"}
	#
	# # 4. 각 구간별 개수 카운트
	# total = len(brand_filtered)
	#
	# top1 = len(brand_filtered[brand_filtered['amt_avg'] >= pct_25])
	# top2 = len(brand_filtered[(brand_filtered['amt_avg'] >= pct_50) & (brand_filtered['amt_avg'] < pct_25)])
	# top3 = len(brand_filtered[(brand_filtered['amt_avg'] >= pct_75) & (brand_filtered['amt_avg'] < pct_50)])

	# 3. brandStats에서 업종 + 월 데이터만 필터
	datSalesAppend_filtered = datSalesAppend[
		(datSalesAppend['uj3_cd'] == uj3_cd) &
		(datSalesAppend['ym_sales'] == pivotYm) &
		(datSalesAppend['brnd_no'] == brnd_no)]

	if datSalesAppend_filtered.empty:
		print("datSalesAppend null")
		return {"error": "브랜드의 해당 월 매출 데이터가 없음"}

	# 4. 각 구간별 개수 카운트
	total = len(datSalesAppend_filtered)

	top1 = len(datSalesAppend_filtered[datSalesAppend_filtered['zone_amt_avg'] >= pct_25])
	top2 = len(datSalesAppend_filtered[(datSalesAppend_filtered['zone_amt_avg'] >= pct_50) & (datSalesAppend_filtered['zone_amt_avg'] < pct_25)])
	top3 = len(datSalesAppend_filtered[(datSalesAppend_filtered['zone_amt_avg'] >= pct_75) & (datSalesAppend_filtered['zone_amt_avg'] < pct_50)])
	top4 = len(datSalesAppend_filtered[datSalesAppend_filtered['zone_amt_avg'] < pct_75])

	# print(total)
	# print(datSalesAppend_filtered)
	# print(pct_25, pct_50, pct_75)
	# print(top1, top2, top3, top4)

	def rate(n) -> float:
		return round(n / total * 100, 1) if total > 0 else None

	return {
		"amt_top1_store_rate": rate(top1),
		"amt_top2_store_rate": rate(top2),
		"amt_top3_store_rate": rate(top3),
		"amt_top4_store_rate": rate(top4)
	}

def getTotalKeywordSearchCount(brndNo: str) -> int:
    row = datKeyword[datKeyword['brnd_no'] == brndNo]

    if row.empty:
        return 0  # 또는 None 반환도 가능

    # 문자열 숫자 → 정수로 변환 (쉼표 제거 포함)
    def toInt(val):
        if isinstance(val, str):
            return int(val.replace(",", ""))
        return int(val)

    keyword = toInt(row.iloc[0]['keyword_cnt'])
    blog = toInt(row.iloc[0]['blog_cnt'])
    cafe = toInt(row.iloc[0]['cafe_cnt'])

    return keyword + blog + cafe

def getYoyOpenCloseStore(brndNo: str, pivotYm: int) -> dict:
    startYm = pivotYm - 100 + 1  # 1년 전 1월부터 포함

    # 1. 필터링
    df = brandStats[
        (brandStats['brnd_no'] == brndNo) &
        (brandStats['ym_sales'] >= startYm) &
        (brandStats['ym_sales'] <= pivotYm)
    ]

    # 2. 결측치 처리 후 합산
    openSum = df['store_open_cnt'].fillna(0).sum()
    closeSum = df['store_close_cnt'].fillna(0).sum()

    return {
        'y_open_store': int(openSum),
        'y_close_store': int(closeSum)
    }

def metaFormatting(meta: Meta) -> Meta:
	meta.brnd_brno = formatBrno(meta.brnd_brno)
	meta.ymd_brnd = formatYmd(meta.ymd_brnd)
	meta.opr_duration = formatMonthsToYM(meta.opr_duration)
	meta.y_total_amt = formatKoreanCurrency(meta.y_total_amt)
	meta.brnd_store_cnt = formatCountUnit(meta.brnd_store_cnt)

	meta.market_amt_rate = formatPercent(meta.market_amt_rate)
	meta.market_store_cnt_rate = formatPercent(meta.market_store_cnt_rate)
	meta.market_amt_rank = formatRankWithTotal(meta.market_amt_rank, meta.market_uj3_total)
	meta.market_store_cnt_rank = formatRankWithTotal(meta.market_store_cnt_rank, meta.market_uj3_total)
	meta.market_uj3_total = formatCountUnit(meta.market_uj3_total)

	meta.y_store_survival = formatMonthsToYM(meta.y_store_survival)
	meta.y_amg_inc_dec_rate = formatPercent(meta.y_amg_inc_dec_rate)
	meta.y_store_cnt_inc_dec_rate = formatPercent(meta.y_store_cnt_inc_dec_rate)
	meta.amt_top1_store_rate = formatPercent(meta.amt_top1_store_rate)
	meta.amt_top2_store_rate = formatPercent(meta.amt_top2_store_rate)
	meta.amt_top3_store_rate = formatPercent(meta.amt_top3_store_rate)
	meta.y_portal_search_cnt = formatCountUnit(meta.y_portal_search_cnt)
	meta.y_open_store = formatCountUnit(meta.y_open_store)
	meta.y_close_store = formatCountUnit(meta.y_close_store)

	print("================ META FORMATTING DATA.. ================")
	print(meta)
	return meta

def metaMain(brnd_no: str, pivotYm: int | str):
	meta = getMetaDefaultData(brnd_no)
	meta.fchhq_nm = getFchhqName(brnd_no)
	meta.opr_duration = getElapsedYearsMonths(meta.ymd_brnd)
	meta.brnd_store_cnt = getAliveStoreCount(brnd_no, pivotYm)
	meta.y_total_amt = getYoYTotalAmt(brnd_no, pivotYm)
	marketBrandStatsDict = marketDataByBranStats(brnd_no, pivotYm)
	addDictToDataClass(meta, marketBrandStatsDict)
	meta.y_store_survival = getAvgStoreLife(brnd_no)
	yoyChangeRateDict = getYoyChangeRate(brnd_no, pivotYm)
	addDictToDataClass(meta, yoyChangeRateDict)
	AmtTopStoreRatesDict = getAmtTopStoreRates(brnd_no, pivotYm)
	addDictToDataClass(meta, AmtTopStoreRatesDict)
	meta.y_portal_search_cnt = getTotalKeywordSearchCount(brnd_no)
	yoyOpenCloseStoreDict = getYoyOpenCloseStore(brnd_no, pivotYm)
	addDictToDataClass(meta, yoyOpenCloseStoreDict)

	print("================ META RAW DATA.. ================")
	print(meta)
	return meta

if __name__ == "__main__":
	metaMain('BRD_20190835', 202412)