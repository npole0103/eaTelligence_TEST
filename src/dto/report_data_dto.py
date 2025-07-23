from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union


@dataclass
class RegionData:
    name: str
    value: int


@dataclass
class StoreTrend:
    xAxisData: List[str]
    yAxisData: List[int]


@dataclass
class SalesTrend:
    xAxisData: List[str]
    yAxisData: List[int]


@dataclass
class AmtRankByCity:
    rank: int
    region: str
    store_count: int


@dataclass
class StoreDataItem:
    rank: int
    name: str
    value: int
    isMe: Optional[bool] = False


@dataclass
class Meta:
    brnd_no: str
    brnd_nm: str
    uj3_nm: str
    fchhq_nm: str
    brnd_rprsv_nm: str
    tel_no: str
    brnd_brno: str
    ymd_brnd: str
    opr_duration: str
    y_total_amt: str

    market_amt_rate: str
    market_store_cnt_rate: str
    market_amt_rank: str
    market_store_cnt_rank: str

    y_store_survival: str
    y_amg_inc_dec_rate: str
    y_store_cnt_inc_dec_rate: str

    amt_top1_store_rate: str
    amt_top2_store_rate: str
    amt_top3_store_rate: str

    y_portal_search_cnt: str
    y_open_store: str
    y_close_store: str


@dataclass
class Page1:
    ai_summary: str
    store_cnt: str
    brandData: List[int]
    averageData: List[int]


@dataclass
class Page2:
    ai_summary: str
    store_cnt: str
    y_open_store: str
    y_close_store: str
    y_diff_store_cnt: str
    opr_duration: str
    regionData: List[RegionData]
    store_trend: StoreTrend


@dataclass
class Page3:
    ai_summary: str
    m_avg_amt: str
    y_avg_amt_per_store: str
    m_avg_amt_per_store: str
    amt_rank_by_city: List[AmtRankByCity]
    sales_trend: SalesTrend


@dataclass
class Page4:
    uj3_store_cnt: str
    uj3_store_cnt_rank: str
    uj3_store_cnt_priority: str
    uj3_amt_rank: str
    uj3_amt_priority: str
    store_data: List[StoreDataItem]
    sales_data: List[StoreDataItem]
    ai_summary: List[str]
    kiosk_rate: str
    tablet_rate: str
    phone_rate: str
    etc_rate: str


@dataclass
class Page5:
    ai_summary: List[str]
    used: str
    unused: str
    keyword_search_cnt: str
    blog_post_cnt: str
    cafe_post_cnt: str
    ai_news: str


@dataclass
class BrandReport:
    meta: Meta
    page1: Page1
    page2: Page2
    page3: Page3
    page4: Page4
    page5: Page5