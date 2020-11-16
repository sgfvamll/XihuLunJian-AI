# -*- coding: utf-8 -*-
import sys
sys.path.append('../train_code')
import numpy as np
import pandas as pd
from utils.utils import *
from train_config import args
from sklearn.neighbors import NearestNeighbors
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def f(x):
    x = pd.Series(x)
    res_v = len(set(x.values))-1
    return res_v

def addtion_func(addtion_path):
    test_path = '../../data/test_1.csv'
    res_path = "../../finalA/S3_finalB.csv"
    encode_type = 'utf-8'
    model_names = [
            'ExtraTreesClassifier', 
            'AdaBoostClassifier', 
            'GradientBoostingClassifier',
            'RandomForestClassifier', 
            'BaggingClassifier',
        ]

    test = pd.read_csv(test_path, keep_default_na=False, encoding=encode_type)
    res = pd.read_csv(res_path, keep_default_na=False, encoding=encode_type)
    test = pd.merge(test, res, on=['eventId'])
    test = test[test['label']==1]
    print(len(test))
    print(test.columns)
    encoders = joblib.load('../train_code//dataSet/encoders.pkl')
    test, _ = ProcessData(test, encoders) 

    drop_cols = args['drop_cols']+['label']
    print(drop_cols)
    x_columns = [x for x in test.columns if x not in drop_cols]

    X = test[x_columns]
    X = pd.DataFrame(encoders[len(encoders)-1].transform(X), columns = x_columns)
    X['srcAddress'] = test['srcAddress'].values

    X = X.groupby(['srcAddress']).agg(['min', 'max', 'mean', 'std'])
    X = X.fillna(0.)

    

    # data = pd.DataFrame(test[['srcAddress', 'destAddress', 'eventId']])
    X = test[x_columns]
    X = pd.DataFrame(encoders[len(encoders)-1].transform(X), columns = x_columns)
    
    SSE = []
    SIS = []
    DIFF = []
    for n_clusters in range(5, 60, 5):
        print("n_cluster:", n_clusters)
        km = KMeans(n_clusters = n_clusters).fit(X)

        labels = km.labels_
        # centroids = km.cluster_centers_

        # list_lablel = list(labels)
        # for i in range(n_clusters):
        #     print(list_lablel.count(i))

        sis = silhouette_score(X, labels, metric='euclidean')
        print("score:", sis, km.inertia_)
        SSE.append(km.inertia_)
        SIS.append(sis)

        # res = pd.DataFrame()
        # res['label'] = km.labels_
        # res['srcAddress'] = test['srcAddress'].values
        
        # res = pd.DataFrame()
        # res['label'] = km.labels_
        # res['srcAddress'] = test['srcAddress'].values
        # diff = (np.array(res.groupby(['srcAddress']).agg(f).values)).sum()
        # DIFF.append(diff)
        # print(np.array(res.groupby(['srcAddress']).agg(f).values))
        # print(res.groupby(['srcAddress']).values)

    X = range(5,20)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X,SSE,'o-')
    plt.show()

    X = range(5,20)
    plt.xlabel('k')
    plt.ylabel('SIS')
    plt.plot(X,SIS,'o-')
    plt.show()

    # X = range(5,20)
    # plt.xlabel('k')
    # plt.ylabel('DIFF')
    # plt.plot(X,DIFF,'o-')
    # plt.show()

    
    # test['label'] = labels
    # submission = test[['eventId', 'label']]
    # submission.to_csv('../../addition/S3_addition.csv', index = False, encoding='utf-8')


if __name__ == '__main__':
    addtion_path = '../../finalA/S3_addtion.csv' # 该路径仅供参考
    addtion_func(addtion_path)

