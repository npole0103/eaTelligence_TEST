from src.common.customFormatter import formatBrno, formatYmd, formatKoreanCurrency, formatMonthsToYM
from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta

from datetime import datetime

datFchhq = load_excel("datFchhq")
datBrnd = load_excel("datBrnd")
brandStats = load_excel("brandStats")
datStore = load_excel("datStore")

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
	"y_total_amt": "y_total_amt",

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

	"y_portal_search_cnt": "y_portal_search_cnt",

	"y_open_store": "y_open_store",
	"y_close_store": "y_close_store"
}

def rowToMeta(row: dict) -> Meta:
    mapped = {
        dc_field: _safe_cast(row.get(pd_col, None))
        for pd_col, dc_field in column_mapping.items()
    }
    return Meta(**mapped)

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
	meta_instance = rowToMeta(datBrnd_filter.iloc[0].to_dict())

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

def getYoYTotalAmt(brnd_no: str) -> str:
	# ① 2024년 1월부터 12월까지 필터링
	df_2024 = brandStats[(brandStats['brnd_no'] == brnd_no)
						 & (brandStats['ym_sales'] >= 202401)
						 & (brandStats['ym_sales'] <= 202412)]
	total_amt = df_2024['amt_avg'].sum()

	return total_amt

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
		'market_amt_rate': f"{amt_avg_pct}%",
		'market_store_cnt_rate': f"{store_cnt_pct}%",
		'market_amt_rank': f"{target_amt_rank} / {total_count} 위",
		'market_store_cnt_rank': f"{target_store_rank} / {total_count} 위"
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

def metaMain(brnd_no: str):
	meta = getMetaDefaultData(brnd_no)
	meta.fchhq_nm = getFchhqName(brnd_no)
	meta.opr_duration = getElapsedYearsMonths(meta.ymd_brnd)
	meta.y_total_amt = getYoYTotalAmt(brnd_no)
	market_brand_stats_dict = marketDataByBranStats(brnd_no, 202412)
	addDictToDataClass(meta, market_brand_stats_dict)
	meta.y_store_survival = getAvgStoreLife(brnd_no)

	meta.brnd_brno = formatBrno(meta.brnd_brno)
	meta.ymd_brnd = formatYmd(meta.ymd_brnd)
	meta.opr_duration = formatMonthsToYM(meta.opr_duration)
	meta.y_total_amt = formatKoreanCurrency(meta.y_total_amt)
	meta.y_store_survival = formatMonthsToYM(meta.y_store_survival)

	print(meta)
	return meta

# BRD_20190835
metaMain('BRD_20190835')

# "market_amt_rate": "T11.2%",
# "market_store_cnt_rate": "T7.4%",
# "market_amt_rank": "T3위",
# "market_store_cnt_rank": "T12위",
#
# "y_store_survival": "T4년 11개월",
# "y_amg_inc_dec_rate": "0.0%",
# "y_store_cnt_inc_dec_rate": "0.0%",
#
# "amt_top1_store_rate": "T53.7%",
# "amt_top2_store_rate": "T46.2%",
# "amt_top3_store_rate": "T0.1%",
#
# "y_portal_search_cnt": "11,524",
#
# "y_open_store": "T1개",
# "y_close_store": "T2개"