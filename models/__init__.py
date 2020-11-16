import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegressionCV

from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

SKLModels = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    LogisticRegressionCV(),
    KNeighborsClassifier(),
    LinearSVC(),
    # XGBClassifier()
]


def get_model(name, PREDICTING=False, path=None):
    if PREDICTING==False:
        model = [model for model in SKLModels if model.__class__.__name__==name]
        try: 
            model = model[0]
        except:
            raise RuntimeError("model \"{}\" not available".format(name))
    else:
        model = load_model(name)
    return model

def get_models(names, PREDICTING=False):
    models = {}
    for name in names:
        models[name] = get_model(name, PREDICTING)
    return models


params_path = "saved_params"

def save_model(model, name, path=None):
    if path!=None:
        joblib.dump(model, os.path.join(params_path, path))
    else:
        joblib.dump(model, os.path.join(params_path, name+'.pkl'))  

def load_model(name, path=None):
    if path!=None:
        model = joblib.load(os.path.join(params_path, path))
    else:
        model = joblib.load(os.path.join(params_path, name+'.pkl'))
    return model
