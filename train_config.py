
args = dict(
    models = [
        {
            "name": 'RandomForestClassifier',
            'params': {
                'n_estimators': [60],
                'max_depth': [12],
                'min_samples_split': [6], 
                'min_samples_leaf': [2],
                # 'max_features': [12],
                'random_state': [123456],
                'class_weight': [{0:1, 1:10}]
            },
        },
        {
            "name": 'ExtraTreesClassifier',
            'params': {
                # 'n_estimators': [70],
                # 'max_depth': [13],
                # 'min_samples_split': [8],
                # 'min_samples_leaf': [2],
                # 'max_features': [13],
                # 'random_state': [123456],
                'class_weight': [{0:1, 1:5}],
            },

        },
        {
            'name': 'AdaBoostClassifier',
            'params': {
                'n_estimators': [400],
                'learning_rate': [0.5],
                'random_state': [123456],
                'algorithm': ['SAMME.R']
            },

        },
        {
            'name': 'GradientBoostingClassifier', 
            'params': {
                'n_estimators': [60],
                'max_depth': [12],
                'min_samples_split': [6],
                'min_samples_leaf': [2],
                'max_features': [0.7],
                'random_state': [123456]
                # fit(X, y, sample_weight=None, monitor=None)
            },


        },
        {
            'name': 'BaggingClassifier',
            'params': {
                'n_estimators': [70],
                'max_features': [0.8],
                'random_state': [123456],
            },

        },
        # {
        #     'name': 'KNeighborsClassifier',
        #     'params': {
        #         'weights': ['distance'],
        #         'algorithm': ['auto'],
        #         'n_neighbors': [3]
        #     },
        # },
        # {
        #     'name':'XGBClassifier',
        #     'params': {

        #     }
        # },
        # {
        #     'name': 'LogisticRegressionCV',
        #     'params': {
        #         'class_weight': ['balanced'],
        #         'solver': ['saga'],
        #         'max_iter': [3000],
        #         'scoring': ['f1'],
        #     }
        # },
    ],
    scoring = "f1",
    cv = 5,
    n_jobs = 6,

    # 设置为True，DataPrepare会重新生成train_train和train_test数据集
    # 在调试、训练特征时使用，
    TRAIN_Features = False,

    # 训练集的比例
    TRAIN_Percent = 0.6,

    # 控制train.py、test.py加载train或test数据还是整个数据
    NEED_TEST = False,

    test_config = {
        "models": [
            'ExtraTreesClassifier', 
            'AdaBoostClassifier', 
            'GradientBoostingClassifier',
            'RandomForestClassifier', 
            'BaggingClassifier',
            # 'KNeighborsClassifier',
            # 'LogisticRegressionCV'
            # 'XGBClassifier'
        ],
        "scoring" : "f1",
        "cv" : 5,
        "n_jobs" : 3,
        'USE_NEURAL_NETWORK': False
    },
    drop_cols = ['srcAddress', 'destAddress', 'tlsSubject', 'appProtocol', 'tlsIssuerDn', 'tlsSni', 'eventId'] + 
                ['C', 'ST', 'L', 'O', 'OU', 'CN'] + 
                ['label'] + 
                ['tls_star', 'tls_XX', 'C_len', 'tls_some_state', 'unknown_len', 'tls_default', 'serialNumber_len'] + 
                ['CN_len', 'ST_len', 'tlsVersion'] + ['ST', 'L', 'O', 'emailAddress', 'serialNumber'],
)
