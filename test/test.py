import os as os
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from common.module import DATA_PATH

def path_test_data():
    cd_uj_path = DATA_PATH / "cd_uj.dat";
    cd_uj = pd.read_csv(cd_uj_path, sep='|', encoding='utf-8');

    print(cd_uj.head());

path_test_data()