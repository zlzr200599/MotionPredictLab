# from torch.autograd import Variable
# from models import MotionPredictLab

# from config import DefaultConfig
# from data import VectorNetDataset, collate
# import torch
# from torch.utils.data import DataLoader
# opt = DefaultConfig()
# torch.multiprocessing.set_sharing_strategy('file_system')

# import dgl
# import dgl.function as fn
# import torch

# g = dgl.heterograph({
#     ('user', 'follows', 'user'): ([0, 1], [1, 1]),
#     ('game', 'attracts', 'user'): ([0], [1])
# })
# g.nodes['user'].data['h'] = torch.tensor([[1.], [2.]])
# g.nodes['game'].data['h'] = torch.tensor([[1.5]])
#
# g.multi_update_all(
#     {
#         'follows': (fn.copy_src('h', 'm'), fn.max('m', 'h')),
#         'attracts': (fn.copy_src('h', 'm'), fn.max('m', 'h')),
#     },
#     "sum"
# )
# print(g.nodes['user'].data['h'])

# a = torch.rand((6, 3, 2))
#
# g = dgl.heterograph({
#     ('track_point', 'point_to_track', 'track_graph'): ([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 1]),
#     ('track_graph', 'track_to_point', 'track_point'): ([0, 0, 0, 0, 0, 1], [0, 1, 2, 3, 4, 5]),
# })
#
# g.nodes['track_point'].data['a'] = a
#
#
# g.update_all(fn.copy_u('a', 'm'), fn.sum('m', 's'), etype="point_to_track")
# print(g.nodes['track_graph'].data['s'])
#
# print(a[:5,:,:].sum(0))
# print(a[5,:,:])
# print("_______________________________________")
# a = torch.rand(1,2,3)
# print(a)
# print(a+torch.tensor([1,2,3]))
# t = torch.arange(0.0, 3.0, 0.1).unsqueeze(dim=1)
# print(t.expand(3,-1,-1).permute(2,0,1))
import os
import pandas as pd
import numpy as np
from anycache import anycache
import torch

a = torch.rand((5, 2))
print(a)
b = torch.tensor([0,0,1,1,0]).unsqueeze(dim=1)
print(b)
c = a*b
print(c)
a = np.array([1,2,3])
print(a)
print(np.concatenate((a,a)))

# av
        av = av.permute(1, 0, 2)
        av_code, (av_h_n, av_c_n) = self.av_encoder(av)
        g.nodes['av'].data['code'] = av_c_n.squeeze(dim=0)
        # others
        others = others.permute(1, 0, 2)
        others_code, (others_h_n, others_c_n) = self.others_encoder(others)
        g.nodes['others'].data['code'] = others_c_n.squeeze(dim=0)
        # lane
        lane = lane.permute(1, 0, 2)
        lane_code, (lane_h_n, lane_c_n) = self.agent_encoder(lane)
        g.nodes['lane'].data['code'] = lane_c_n.squeeze(dim=0)
