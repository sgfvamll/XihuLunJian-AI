import sys
sys.path.append('../train_code')
import numpy as np
import pandas as pd
from utils.utils import *
from train_config import args
from sklearn.neighbors import NearestNeighbors
import joblib

def addtion_func(addtion_path):
    test_path = '../../data/test_1.csv'
    res_path = "../../finalA/S3_finalA.csv"
    encode_type = 'utf-8'

    test = pd.read_csv(test_path, keep_default_na=False, encoding=encode_type)
    res = pd.read_csv(res_path, keep_default_na=False, encoding=encode_type)
    test = pd.merge(test, res, on=['eventId'])
    test = test[test['label']==1]

    encoders = joblib.load('../train_code//dataSet/encoders.pkl')
    test, _ = ProcessData(test, encoders) 

    drop_cols = args['drop_cols']+['label']
    # print(drop_cols)
    x_columns = [x for x in test.columns if x not in drop_cols]

    X = test[x_columns]
    X = pd.DataFrame(encoders[len(encoders)-1].transform(X), columns = x_columns)
    X['srcAddress'] = test['srcAddress'].values

    X = X.groupby(['srcAddress']).agg(['min', 'max', 'mean', 'std'])
    X = X.fillna(0.)

    from sklearn.cluster import KMeans
    n_clusters = 30
    km = KMeans(n_clusters = n_clusters).fit(X)
    labels = km.labels_

    X['label'] = km.labels_
    X = X['label'].reset_index()
    res = pd.merge(test[['eventId', 'srcAddress']], X, on=['srcAddress'])

    res[['eventId', 'label']].to_csv(addtion_path, index = False, encoding='utf-8')



if __name__ == '__main__':
    addtion_path = '../../addition/S3_addition.csv' # 该路径仅供参考
    addtion_func(addtion_path)


