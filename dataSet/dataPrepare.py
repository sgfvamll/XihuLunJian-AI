from utils.utils import *
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os 
import joblib
from train_config import args


print(__name__)

path = "../../data/"
train = None 
test = None 
data = None
features = None

TRAIN_Percent = args['TRAIN_Percent']
TRAIN_Features = args['TRAIN_Features']

if TRAIN_Features or not os.path.exists(os.path.join(path, 'train_train.csv')):
    print("Read train")
    data_path = os.path.join(path, 'train.csv')
    data = pd.read_csv(data_path, keep_default_na=False, encoding="utf-8")
    number = len(data)

    # Divide data into two classes randomly
    data = shuffle(data, random_state=0)
    train = pd.DataFrame(data[:int(number*TRAIN_Percent)])
    test = pd.DataFrame(data[int(number*TRAIN_Percent):])

    ext_info = data[['srcAddress', 'destAddress', 'eventId']]

    # Extract the features. 
    train, encoders = ProcessData(train)
    test, _ = ProcessData(test, encoders)

    # Drop the unuseless cols. 
    drop_cols = args['drop_cols']
    xy_columns = [x for x in train.columns if x not in drop_cols]


    train_y = train['label']
    train = train[xy_columns]

    test_y = test['label']
    test = test[xy_columns]

    # scaler the data to fit linear classier. 
    scaler = StandardScaler(copy=False)
    train = pd.DataFrame(scaler.fit_transform(train), columns = xy_columns)
    test = pd.DataFrame(scaler.transform(test), columns = xy_columns)
    encoders.append(scaler)

    # 做PCA后，Test: 0.984829 -> 0.975980 ... 
    # pca = PCA(n_components="mle", random_state=10)
    # train = pd.DataFrame(pca.fit_transform(train))
    # test = pd.DataFrame(pca.transform(test))
    # encoders.append(pca)

    train['label'] = list(train_y.astype(int).values)
    test['label'] = list(test_y.astype(int).values)

    data = pd.concat([train, test])

    features = data.columns[data.columns!='label']

    # save the data for futher use. 
    train.to_csv(os.path.join(path, "train_train.csv"), index=False)
    test.to_csv(os.path.join(path, "train_test.csv"), index=False)
    ext_info.to_csv(os.path.join(path, "ext_info.csv"), index=False)
    
    joblib.dump(encoders, "./dataSet/encoders.pkl")

else:
    print("Read train_train")
    train = pd.read_csv(os.path.join(path, "train_train.csv"), keep_default_na=False, encoding="utf-8")
    test  = pd.read_csv(os.path.join(path, "train_test.csv"), keep_default_na=False, encoding="utf-8")
    ext_info  = pd.read_csv(os.path.join(path, "ext_info.csv"), keep_default_na=False, encoding="utf-8")
    data = pd.concat([train, test])
    features = data.columns[data.columns!='label']
