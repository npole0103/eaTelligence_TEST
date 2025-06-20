import os as os
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import common.module

# 글로벌 변수
ROOT_PATH = Path.cwd()
DATA_PATH = ROOT_PATH / 'dat'