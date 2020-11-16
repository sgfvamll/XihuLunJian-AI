# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from train import train_start


def train_func(train_path):
    train_start()
    pass

if __name__ == '__main__':
    train_path = '../data/train.csv' # 该路径仅供参考
    train_func(train_path)
