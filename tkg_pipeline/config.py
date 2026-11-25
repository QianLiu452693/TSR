"""参数与路径。数值与研究方案中的清洗、对齐、特征工程设定一致。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COUNTY_DIR = DATA_DIR / "county_geo"
OUTPUT_DIR = ROOT / "output"
SOURCE_CSV = ROOT / "ChinaCrimeDatas.csv"
PCA_JSON = DATA_DIR / "pca-code.json"

# 数据集公开说明中的采集区间（Figshare 28106939 / Scientific Data 2025）
TIME_START = "2000-01-01T00:00:00+08:00"
TIME_END = "2019-12-31T23:59:59+08:00"

# 中国陆海疆域粗边界（含南海诸岛与黑龙江北端），用于规则剔除
LON_MIN, LON_MAX = 73.4, 135.1
LAT_MIN, LAT_MAX = 3.8, 53.6

# DBSCAN 孤立点：3 km 邻域内没有其他案件视为噪声
DBSCAN_EPS_M = 3000.0
DBSCAN_MIN_SAMPLES = 2

# 无案件编号时的三重查重
DEDUP_DISTANCE_M = 300.0
DEDUP_TIME_SECONDS = 2 * 3600
DEDUP_JACCARD = 0.5

# 城市网格
GRID_SIZE_M = 500
GRID_INDEX_MAX = 999

# 时空相继
SUCCESSION_SECONDS = 24 * 3600

# 罪名语义匹配
TYPE_ACCEPT_SCORE = 0.35

# 关系置信度
RULE_CONFIDENCE = 1.0
SUCCESSION_CONFIDENCE = 0.98
LOW_CONFIDENCE = 0.30
CAUSAL_MIN_LIFT = 1.5
CAUSAL_COUNTY_MIN_COUNT = 20
CAUSAL_TYPE_MIN_COUNT = 30
CAUSAL_GRID_MIN_COUNT = 50
CAUSAL_GRID_MIN_LIFT = 3.0

REVIEW_FRACTION = 0.01
RANDOM_SEED = 42

SOURCE_COLUMNS = [
    "case_number",
    "case_type",
    "city",
    "latitude",
    "longitude",
    "court_name",
    "incident_location",
    "incident_province",
    "incident_city",
    "incident_county",
    "incident_time",
    "formatted_datetime",
    "judgment_date",
    "details",
]
