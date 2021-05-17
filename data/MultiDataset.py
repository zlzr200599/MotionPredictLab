from abc import ABC
import torch as th
import dgl
import time
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import pandas as pd
# from dgl.data import DGLDataset
# from torch.utils.data import DataLoader,Dataset
from dgl.dataloading import GraphDataLoader
import os
import numpy as np
# import os
from typing import List, Union, Tuple, Callable, Dict
# from pathlib import Path
# from typing import Any, Sequence
from dgl.data import DGLDataset
from functools import lru_cache
from anycache import anycache, AnyCache

argo_map = ArgoverseMap()
argo_center_lines = argo_map.city_lane_centerlines_dict


def get_lane_center_lines(center_coordinate: Union[Tuple[float, float], np.ndarray],
                          radius: float,
                          city_name: str) -> List[np.ndarray]:
    center_x, center_y = center_coordinate
    up, down, left, right = center_y + radius, center_y - radius, center_x - radius, center_x + radius
    center_lines_list = []
    for lane_id, lane_seg in argo_center_lines[city_name].items():
        # lane_seg has attributes :
        #  has_traffic_control,turn_direction,is_intersection,l_neighbor_id,r_neighbor_id,predecessors,successors,centerline,
        curr_center_line = lane_seg.centerline
        mid_idx = curr_center_line.shape[0] // 2
        x, y = curr_center_line[mid_idx, :]
        if left < x < right and down < y < up:
            center_lines_list.append(curr_center_line)
    return center_lines_list


def dict_to_graph(input_dict: Dict):
    x, y = input_dict['radix']['x'], input_dict['radix']['y']
    center = np.array([x, y])
    agent_track = input_dict['agent'] - center
    start_time = input_dict['start_time']
    time_dict = input_dict['time_dict']
    n_avs, n_others, n_lanes = len(input_dict['av']), len(input_dict['others']), len(input_dict['lanes'])

    av_tracks = np.zeros((n_avs, 50, 2), np.float)
    av_tracks_len = np.zeros((n_avs, 2), np.int)
    av_tracks_mask = np.zeros((n_avs, 50, 2), np.int)
    for idx, t_array in enumerate(input_dict['av'].values()):
        t_start = t_array[0, 0]
        i_start = time_dict[t_start]
        av_tracks[idx, i_start: i_start+len(t_array), :] = t_array[:, 1:]  - center
        av_tracks_len[idx, :] = np.array([i_start, i_start + len(t_array)], dtype=np.int)
        av_tracks_mask[idx, i_start: i_start+len(t_array), :] = np.zeros_like(t_array[:, 1:])
    other_tracks = np.zeros((n_others, 50, 2), np.float)
    other_tracks_len = np.zeros((n_others, 2), np.int)
    other_tracks_mask = np.zeros((n_others, 50, 2), np.int)
    for idx, t_array in enumerate(input_dict['others'].values()):
        t_start = t_array[0, 0]
        i_start = time_dict[t_start]
        other_tracks[idx, i_start: i_start + len(t_array), :] = t_array[:, 1:] - center
        other_tracks_len[idx, :] = np.array([i_start, i_start + len(t_array)], dtype=np.int)
        other_tracks_mask[idx, i_start: i_start + len(t_array), :] = np.ones_like(t_array[:, 1:])
    lane_lines = np.zeros((n_lanes, 10, 2), np.float)
    for idx, t_array in input_dict['lanes'].items():
        lane_lines[idx, :, :] = t_array - center

    graph = dgl.heterograph({
        ('agent', 'agent_env', 'env'): ([0], [0]),
        # ('env', 'env_agent', 'agent'): ([0], [0]),

        ('av', 'av_env', 'env'): (list(range(n_avs)), [0]*n_avs),
        # ('env', 'env_av', 'av'): ([0]*n_avs, list(range(n_avs))),

        ('others', 'others_env', 'env'): (list(range(n_others)), [0]*n_others),
        # ('env', 'env_others', 'others'): ([0] * n_others, list(range(n_others))),

        ('lane', 'lane_env', 'env'): (list(range(n_lanes)), [0]*n_lanes),
    })

    graph.nodes['agent'].data['state'] = th.tensor(agent_track, dtype=th.float).unsqueeze(dim=0)
    graph.nodes['av'].data['state'] = th.tensor(av_tracks, dtype=th.float)
    graph.nodes['av'].data['len'] = th.tensor(av_tracks_len, dtype=th.float)
    graph.nodes['av'].data['mask'] = th.tensor(av_tracks_mask, dtype=th.float)
    graph.nodes['others'].data['state'] = th.tensor(other_tracks, dtype=th.float)
    graph.nodes['others'].data['len'] = th.tensor(other_tracks_len, dtype=th.float)
    graph.nodes['others'].data['mask'] = th.tensor(other_tracks_mask, dtype=th.float)
    graph.nodes['lane'].data['state'] = th.tensor(lane_lines, dtype=th.float)

    return graph


def collate(samples):
    batched_graph, info_dict_list = map(list, zip(*samples))
    return dgl.batch(batched_graph), info_dict_list

class MyDataset(DGLDataset, ABC):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    split data set : float
        proportion of the dataset
    """

    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 fraction=1.0,
                 mode="train",
                 ):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        self.ac = AnyCache(cachedir=save_dir)
        self.argo_loader = ArgoverseForecastingLoader(raw_dir)
        self.fraction = fraction if fraction <= 1.0 else 1.0
        self.mode = mode

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    @lru_cache(maxsize=250000)
    def __getitem__(self, idx):
        # get one example by index
        @self.ac.anycache()
        def idx_to_graph(sample_id: int):
            my_dict = {}
            argo_sample = self.argo_loader[idx]
            seq_df = argo_sample.seq_df
            if self.mode == 'test':
                my_dict['df'] = seq_df
            # timestamp_iter = map(lambda t: round(float(t), 1), np.unique(seq_df["TIMESTAMP"].values).tolist())

            track_id_list = argo_sample.track_id_list

            my_dict['city'] = argo_sample.city
            my_dict['filename'] = os.path.splitext(os.path.basename(argo_sample.current_seq))[0]
            my_dict['timestamp'] = np.unique(seq_df["TIMESTAMP"].values).tolist()
            my_dict['time_dict'] = {t: idx for idx, t in enumerate(my_dict['timestamp'])}
            my_dict['start_time'], my_dict['end_time'] = my_dict['timestamp'][0], my_dict['timestamp'][-1]
            my_dict['split_time'] = my_dict['timestamp'][19]
            my_dict['agent'] = argo_sample.agent_traj
            my_dict['radix'] = {'x': my_dict['agent'][19][0], 'y': my_dict['agent'][19][1], 'yaw': 0, }
            my_dict['av'] = {}
            my_dict['agent_track_id'] = ""
            for track_id, row in seq_df.iterrows():
                if str(row['OBJECT_TYPE']) == "AGENT":
                    my_dict['agent_track_id'] = row['TRACK_ID']
                    break
            for track_id in track_id_list:
                track = seq_df[(seq_df["TRACK_ID"] == track_id) & (seq_df["OBJECT_TYPE"] == "AV")][
                    ["TIMESTAMP", "X", "Y"]].to_numpy()
                if len(track) > 0:
                    my_dict['av'][track_id] = track

            my_dict['others'] = {}
            for track_id in track_id_list:
                track = seq_df[(seq_df["TRACK_ID"] == track_id) & (seq_df["OBJECT_TYPE"] == "OTHERS")][
                    ["TIMESTAMP", "X", "Y"]].to_numpy()
                if len(track) > 0:
                    my_dict['others'][track_id] = track

            my_dict['lanes'] = {
                lane_id: lane_seq[:10, :]
                if len(lane_seq) >= 10
                else np.concatenate((lane_seq, np.zeros((10-len(lane_seq), 2))
                                     ))
                for lane_id, lane_seq in enumerate(get_lane_center_lines(center_coordinate=my_dict['agent'][19],
                                                                         radius=50,
                                                                         city_name=my_dict['city']
                                                                         ))
            }
            return my_dict
        my_dict = idx_to_graph(sample_id=idx)
        return dict_to_graph(my_dict), my_dict

    def __len__(self):
        # number of data examples
        return int(self.fraction * len(self.argo_loader))

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def clear_cache(self):
        # clear the processed data from directory 'self.save_path'
        print('clear the cache')
        self.ac.clear()

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

    @property
    def cache_size(self):
        return self.ac.size

class AllDataset:
    def __init__(self,
                 train_dir=None,
                 train_fraction=1.0,
                 val_dir=None,
                 val_fraction=1.0,
                 test_dir=None,
                 test_fraction=1.0,
                 ):
        self.train = MyDataset(raw_dir=train_dir, fraction=train_fraction, mode='train')
        self.val = MyDataset(raw_dir=val_dir, fraction=val_fraction, mode='val')
        self.test = MyDataset(raw_dir=test_dir, fraction=test_fraction, mode='test')

if __name__ == "__main__":
    raw_dir = './train/train_1k'
    raw_dir = './train/train_1h'
    dataset = MyDataset(raw_dir=raw_dir, fraction=1.01, save_dir='./tmp/any.my')
    start_time = time.time()
    data_loader = GraphDataLoader(dataset, collate_fn=collate, batch_size=16, shuffle=False, drop_last=True)
    start = time.time()
    for _ in range(2):
        print(f"epoch : {_}-------------------------------------------------------------------")
        for i, (bhg, info) in enumerate(data_loader):
            print(f"{i} | {time.time() - start_time:6.2f} s")
        print(f"current time: {time.time()-start:6.2f} s")
    print(dataset.cache_size)
    dataset.clear_cache()