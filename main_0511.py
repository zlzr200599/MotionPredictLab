# import torch
# import dgl
# import time
from data.mydataset import AllDataset
# import os
# import numpy as np
import warnings
from models.My_Model_0511 import MyModel

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # train_dir = './data/test/test_sample'
    # train_dir = './data/train/forecasting_val_v1.1'
    # train_dir = './data/train/forecasting_train_head_1000'
    train_dir = './data/train/forecasting_train_head_10000'
    # test_dir = '/home/huanghao/Lab/argodataset/test_obs/data'  # 78143
    # test_dir = './data/test/test_sample'
    # test_dir = './data/train/forecasting_val_v1.1'  # 200
    test_dir = '/home/huanghao/Lab/MotionPredictLab/data/test/jd_2000'
    data = AllDataset(train_dir=train_dir,
                      train_fraction=0.250,
                      val_dir='/home/huanghao/Lab/argodataset/train/data',
                      val_fraction=0.00050,  # 108012
                      test_dir=test_dir,
                      test_fraction=1.0,
                      )

    model = MyModel()
    model.train_model(dataset=data, batch_size=16, shuffle=True,
                      n_epoch=10, lr=0.05
                      )
    model.load('new.pth')
    model.test_model(dataset=data, output_dir="./jd_result/")
