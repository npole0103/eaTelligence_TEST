from src.db.load_db_csv import load_excel
from src.dto.report_data_dto import Meta

# BRD_20190835
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

def row_to_meta(row: dict) -> Meta:
    mapped = {
        dc_field: row.get(pd_col, None)  # get()만 쓰면 충분
        for pd_col, dc_field in column_mapping.items()
    }
    return Meta(**mapped)

def getMetaDefaultData(brnd_no: str) -> Meta:
	datBrnd = load_excel("datBrnd")

	datBrnd = datBrnd[datBrnd['brnd_no'] == brnd_no]
	meta_instance = row_to_meta(datBrnd.iloc[0].to_dict())

	return meta_instance

meta = getMetaDefaultData('BRD_20190835')
print(meta)