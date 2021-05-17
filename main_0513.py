
# import time
from data.mydataset import AllDataset
# import os
# import numpy as np
import warnings
from models.My_Model_0513 import MyModel

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    train_dir = './data/train/train_10k'
    test_dir = './data/test/test_2k'

    data = AllDataset(train_dir=train_dir,
                      train_fraction=1.250,
                      val_dir='/home/huanghao/Lab/argodataset/val/data',
                      val_fraction=2000/39472,  # 39472
                      test_dir=test_dir,
                      test_fraction=1.0,
                      )

    model = MyModel(saved_path='new_20210517_0000.pth')
    # model.train_model(dataset=data, batch_size=16, shuffle=True,
    #                   n_epoch=100, lr=0.05,
    #                   )
    model.val_model(dataset=data, return_to_plot=False)
    # model.test_model(dataset=data, output_dir="./test_result_0514_01/")
