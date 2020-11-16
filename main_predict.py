# -*- coding: utf-8 -*-
# from dataSet.dataPrepare import ProcessData
from models import get_models
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from utils.utils import *
import joblib
from train_config import args



def test_func(test_path, save_path):
    model_names = [
                'ExtraTreesClassifier', 
                'AdaBoostClassifier', 
                'GradientBoostingClassifier',
                'RandomForestClassifier', 
                'BaggingClassifier',
            ]
    encode_type = 'utf-8'
    test = pd.read_csv(test_path, keep_default_na=False, encoding=encode_type)
    encoders = joblib.load('./dataSet/encoders.pkl')
    
    # 数据预处理
    test, _ = ProcessData(test, encoders) 

    # 扔掉不用的标签字段（例如：值类型为字符串的字段）
    drop_cols = args['drop_cols']
    x_columns = [x for x in test.columns if x not in drop_cols]
    X = test[x_columns]

    # 提取一些信息用于后处理
    data = pd.DataFrame(test[['srcAddress', 'destAddress', 'eventId']])


    # scale
    X = pd.DataFrame(encoders[len(encoders)-1].transform(X), columns = x_columns)
    
    # 在多个模型上分别预测
    models = get_models(model_names, True)
    for model_name in model_names:
        y_pred = models[model_name].predict(X)
        data[model_name] = y_pred

    # 统计投票结果
    y_preds = np.sum(data[models].to_numpy(), axis=1)
    # y_preds = (y_preds > (len(models)//2)) + 0
    data['y_preds'] = y_preds

    # 后处理；认为同一组['srcAddress','destAddress']的标签结果应该相同，故而通过均值聚合预测结果
    data['index']=list(range(len(data)))
    data=data.merge(
        data[['srcAddress','destAddress','y_preds']].groupby(
                ['srcAddress','destAddress'], as_index=False
            ).mean(), 
        on=['srcAddress','destAddress'], 
        suffixes=['', '_mean_by_addr']
    ).sort_values('index')

    # 加合同组均值预测结果与单样本预测结果
    result = data[['y_preds'+'y_preds_mean_by_addr']]
    y_preds = np.sum(result.to_numpy(), axis=1)
    y_preds = (y_preds > (len(models))) + 0
    data['label'] = y_preds

    submission = data[['eventId', 'label']]
    submission.to_csv(save_path + 'S3_finalB.csv',index = False, encoding='utf-8')

if __name__ == '__main__':
    test_path = './data/test_1.csv'
    sava_path = './'
    test_func(test_path,sava_path)

