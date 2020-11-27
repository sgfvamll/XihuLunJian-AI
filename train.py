import numpy as np
from models import get_models, save_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import joblib
import pandas as pd
import random


def train_start():
    from train_config import args
    from dataSet.dataPrepare import train, test, data, features, ext_info
    print(features, len(features))
    if args['NEED_TEST']:
        X = train[features]
        y = train['label']
    else:
        X = data[features]
        y = data['label']

    print("Samples Number: ", len(y))

    models = get_models([model['name'] for model in args['models']], False)


    def myfit(model, gsearch, X, y):
        if model['name'] in ['AdaBoostClassifier', 'GradientBoostingClassifier']:
            weights = [5 if x == 1 else 1 for x in list(y.values)]
            gsearch.fit(X, y, sample_weight=weights)
        else:
            gsearch.fit(X, y)


    for model in args['models']:
        gsearch= GridSearchCV(
            estimator = models[model['name']],
            param_grid =model['params'], 
            scoring=args['scoring'],
            cv=args['cv'],
            n_jobs=args['n_jobs'],
            refit=True
        )

        myfit(model, gsearch, X, y)

        with open("result.train", 'a') as f:
            f.write(str(gsearch.cv_results_)+"\n")
        print(model['name']+":", gsearch.best_score_)
        try:
            imps = gsearch.best_estimator_.feature_importances_
            cols = X.columns
            with open("result.model.imp", 'a') as f:
                for idx in range(len(cols)):
                    f.write(cols[idx]+":", imps[idx])
        except:
            pass
        save_model(gsearch.best_estimator_, model['name'])
        


    print("--------Test-----------")
    if args['NEED_TEST']:
        X = test[features]
        y = test['label']

        args = args['test_config']
        models = get_models([model for model in args['models']], True)

        data = pd.DataFrame(ext_info[-len(y):])

        for model_name in args['models']:
            y_pred = models[model_name].predict(X)
            print(model_name+": F1 Score (Test): %f" % metrics.f1_score(y, y_pred.astype(np.int0)))
            data[model_name] = y_pred


        y_preds = np.sum(data[args['models']].to_numpy(), axis=1)
        data['y_preds'] = y_preds
        y_preds = (y_preds > (len(models)//2)) + 0
        print('--------------')
        print("F1 Score vote (Test): %f" % metrics.f1_score(y, y_preds))

        # Grouped by addresses and calculate the mean of `y_preds`
        data['index']=list(range(len(data)))
        data=data.merge(
            data[['srcAddress','destAddress','y_preds']].groupby(
                    ['srcAddress','destAddress'], as_index=False
                ).mean(), 
            on=['srcAddress','destAddress'], 
            suffixes=['', '_mean_by_addr']
        ).sort_values('index')


        result = data[args['models']+['y_preds_mean_by_addr']]

        y_preds = np.sum(result.to_numpy(), axis=1)
        # mask = np.logical_or(y_preds, )
        # print(y_preds[:20], len(models))
        y_preds = (y_preds >= (len(models))) + 0
        print('--------------')
        print("F1 Score vote with addresses (Test): %f" % metrics.f1_score(y, y_preds))
        print('--------------')

        # mask = np.array(y_preds != y) 
        # # print(mask)
        # data.iloc[mask].to_csv("./error.csv")

