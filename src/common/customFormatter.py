from datetime import datetime

def formatKoreanCurrency(value: int | str) -> str:
	try:
		# 문자열이면 숫자로 변환
		thousand_won = int(value)
	except (ValueError, TypeError):
		return "유효하지 않은 값"

	won = thousand_won * 1000  # 천원 → 원

	if won >= 100_000_000:
		amount = won / 100_000_000
		return f"{amount:.2f}억원"
	elif won >= 10_000:
		amount = won // 10_000
		return f"{amount}만원"
	else:
		return f"{thousand_won}천원"

def formatBrno(brno: int | str) -> str:
    brno_str = str(brno).zfill(10)  # 10자리로 맞춤 (앞에 0 채움)
    return f"{brno_str[:3]}-{brno_str[3:8]}-{brno_str[8:]}"

def formatYmd(ymd: int | str) -> str:
    date = datetime.strptime(str(ymd), "%Y%m%d")
    return date.strftime("%Y년 %m월 %d일")

def formatMonthsToYM(months: int) -> str:
    years = months // 12
    remaining_months = months % 12
    return f"{years}년 {remaining_months}개월"

def formatPercent(data: int | str) -> str:
	if data is None:
		return "0.0%"

	return f"{data:.1f}%"

def formatCountUnit(count: int | None) -> str:
    if count is None:
        return "0개"
    return f"{count}개"

def formatRankWithTotal(rank: int, total: int) -> str:
    return f"{rank}/{total} 위"

def formatRank(rank: int) -> str:
    return f"{rank}위"