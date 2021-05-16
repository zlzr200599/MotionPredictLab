from abc import ABC
import torch as th
import dgl
import time
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import pandas as pd
# from dgl.data import DGLDataset
# from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
# import os
from typing import List, Union, Tuple, Callable
# from pathlib import Path
# from typing import Any, Sequence
from dgl.data import DGLDataset


def get_center_agent_traj(argo_loader_obj: "ArgoverseForecastingLoader") -> np.ndarray:
    """ extract the center agent trajectory line
    Args:
    argo_loader_obj : ArgoverseForecastingLoader object
    Return;
    numpy.ndarray for center agent trajectory track coordinates
    """
    return argo_loader_obj.agent_traj



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


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def calc_transform_radix(center_trajectory, obs_ending_idx=19):
    center_coordinate = center_trajectory[obs_ending_idx]
    sample_frequency = 10  # sample time interval 0.1 second
    velocity, pre_velocity = sample_frequency * (
            center_trajectory[obs_ending_idx:obs_ending_idx - 2:-1]
            - center_trajectory[obs_ending_idx - 1:obs_ending_idx - 3:-1]
    )
    rho, phi = cart2pol(*velocity)
    pre_rho, pre_phi = cart2pol(*pre_velocity)
    line_speed, yaw = rho, phi
    angular = phi - pre_phi
    if angular > np.pi:
        angular = 2 * np.pi - angular
    if angular < -np.pi:
        angular = 2 * np.pi + angular
    angular_velocity = sample_frequency * angular
    return center_coordinate, velocity, yaw, line_speed, angular_velocity


def track_transform_wrapper(agent_trajectory):
    center_coordinate, velocity, yaw, line_speed, angular_velocity = calc_transform_radix(agent_trajectory)

    # for i in range(19, 40):
    #     print(calc_transform_radix(agent_trajectory, i))

    def state_transform_func(state: np.ndarray):
        x, y = state - center_coordinate
        rho, phi = cart2pol(x, y)
        return np.array([x, y, rho, phi - yaw])

    def transform(trajectory):
        return np.apply_along_axis(state_transform_func, axis=1, arr=trajectory)

    return transform


def build_one_track_graph(traj: np.ndarray, transform_func: Callable[[np.ndarray], Callable], n_obs=20):
    # g = dgl.heterograph({
    #     ('track_point', 'point_to_track', 'track_graph'): (list(range(n_obs)), [0]*n_obs),
    #     ('track_graph', 'track_to_point', 'track_point'): ([0]*n_obs, list(range(n_obs))),
    # })
    trans_track = transform_func(traj)
    # g.nodes['track_point'].data["state"] = th.tensor(trans_track)[:n_obs]
    g_feats = th.tensor(trans_track, dtype=th.float)[:n_obs]
    target_feats = th.tensor(trans_track, dtype=th.float)[n_obs:, :2].flatten()
    return g_feats, target_feats


def lane_transform_wrapper(agent_trajectory, end_idx=19):
    center = agent_trajectory[end_idx]

    def transform(lane):
        return lane - center

    return transform


def build_one_lane_graph(lane: np.ndarray, transform_func: Callable[[np.ndarray], Callable]):
    # num_points = lane.shape[0]
    # g = dgl.heterograph({
    #     ('lane_point', 'point_to_lane', 'lane_graph'): (list(range(num_points)), [0]*num_points),
    #     ('lane_graph', 'lane_to_point', 'lane_point'): ([0]*num_points, list(range(num_points))),
    # })
    trans_track = transform_func(lane)
    # g.nodes['lane_point'].data["state"] = th.tensor(trans_track)[:num_points]
    return th.tensor(trans_track, dtype=th.float)


def get_other_tracks(input_df: pd.DataFrame, track_id_list:List[int]) -> List[Tuple[np.ndarray, int]]:
    """Get the trajectory for the track of type 'OTHERS' AND 'AV' in the current sequence.

    Returns:
        list len = num_object, each has two fields(obs, target)
    """
    split_time_stamp = np.unique(input_df["TIMESTAMP"].values)[19]
    others_list = []
    for track_id in track_id_list:
        track = input_df[(input_df["TRACK_ID"] == track_id) & (input_df["OBJECT_TYPE"] != "AGENT")][["TIMESTAMP", "X", "Y"]]
        if len(track) > 0:
            obs = track[track["TIMESTAMP"] <= split_time_stamp][["X","Y"]].to_numpy()
            target = track[track["TIMESTAMP"] > split_time_stamp][["X","Y"]].to_numpy()
            if len(obs) > 0:
                others_list.append((np.concatenate((obs, target)), len(obs)))
        # agent_traj = np.column_stack((agent_x, agent_y))
    return others_list

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
                 fraction=1.0
                 ):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

        self.argo_loader = ArgoverseForecastingLoader(raw_dir)
        self.fraction = fraction if fraction <= 1.0 else 1.0

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __getitem__(self, idx):
        # get one example by index
        curr_argo_sample = self.argo_loader[idx]
        seq_df = curr_argo_sample.seq_df
        others = get_other_tracks(seq_df, curr_argo_sample.track_id_list)
        # print("obs_other")
        # print(len(obs_others))
        # print(obs_others)
        # print([len(o) for o in obs_others])
        # print("target_others")
        # print(len(target_others))
        # print(target_others)
        # print([len(o) for o in target_others])
        curr_agent_traj = get_center_agent_traj(curr_argo_sample)
        curr_lane_center_lines = get_lane_center_lines(center_coordinate=curr_agent_traj[19, :],
                                                       radius=50,
                                                       city_name=curr_argo_sample.city)
        all_obs = [(curr_agent_traj, 20)] + others
        track_feats = [
            build_one_track_graph(traj=tmp_track,
                                  transform_func=track_transform_wrapper(curr_agent_traj),
                                  n_obs=tmp_obs
                                  )
            for tmp_track, tmp_obs in all_obs  #
        ]
        obs_feats, target_feats = map(list, zip(*track_feats))
        lane_feats = [
            build_one_lane_graph(lane=tmp_lane, transform_func=lane_transform_wrapper(curr_agent_traj))
            for tmp_lane in curr_lane_center_lines
        ]

        track_point_list, track_graph_list, lane_point_list, lane_graph_list = [], [], [], []

        tmp_sum = 0
        track_state = []
        for idx, line in enumerate(obs_feats):
            n_line = len(line)
            track_point_list.extend(list(range(tmp_sum, tmp_sum + n_line)))
            track_graph_list.extend([idx] * n_line)
            track_state.append(line)
            tmp_sum += n_line

        target_mask = []
        target_padding = []
        for idx, line in enumerate(target_feats):
            n_line = len(line)
            target_mask.append(th.tensor([[1.0] * n_line + [0.0] * (60 - n_line)]))
            f = th.zeros((1, 60))
            f[0, :n_line] = line
            target_padding.append(f)

        tmp_sum = 0
        lane_state = []
        for idx, line in enumerate(lane_feats):
            n_line = len(line)
            lane_point_list.extend(list(range(tmp_sum, tmp_sum + n_line)))
            lane_graph_list.extend([idx] * n_line)
            lane_state.append(line)
            tmp_sum += n_line

        g = dgl.heterograph({
            ('track_point', 'point_to_track', 'track_graph'): (track_point_list, track_graph_list),
            ('track_graph', 'track_to_point', 'track_point'): (track_graph_list, track_point_list),
            ('lane_point', 'point_to_lane', 'lane_graph'): (lane_point_list, lane_graph_list),
            ('lane_graph', 'lane_to_point', 'lane_point'): (lane_graph_list, lane_point_list),
            ('track_graph', 'track_to_track', 'track_graph'): (
                list(range(len(obs_feats)))*len(obs_feats),
                [i for i in range(len(obs_feats)) for _ in range(len(obs_feats))]
            ),
            ('lane_graph', 'lane_to_lane', 'lane_graph'): (
                list(range(len(lane_feats))) * len(lane_feats),
                [i for i in range(len(lane_feats)) for _ in range(len(lane_feats))]
            ),
            ('track_graph', 'track_to_lane', 'lane_graph'): (
                list(range(len(obs_feats))) * len(lane_feats),
                [i for i in range(len(lane_feats)) for _ in range(len(obs_feats))]
            ),
            ('lane_graph', 'lane_to_track', 'track_graph'): (
                list(range(len(lane_feats))) * len(obs_feats),
                [i for i in range(len(obs_feats)) for _ in range(len(lane_feats))]
            ),
        })
        g.nodes['track_point'].data['state'] = th.cat(track_state)
        # print(lane_state)
        # print(len(lane_state))
        g.nodes['lane_point'].data['state'] = th.cat(lane_state)
        transform_radix = curr_agent_traj
        base_name = os.path.basename(curr_argo_sample.current_seq)
        seq_id = int(os.path.splitext(base_name)[0])
        return g, th.cat(target_padding), th.cat(target_mask), transform_radix, curr_argo_sample.city, seq_id

    def __len__(self):
        # number of data examples
        return int(self.fraction * len(self.argo_loader))

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass


def collate(samples):
    batched_graph, batched_target, batched_av_mask, batched_transform_radix, city, seq_id = map(list, zip(*samples))
    n_target_of_each_sample = [len(s) for s in batched_target]
    agent_index = [0]
    for idx in range(len(n_target_of_each_sample) - 1):
        agent_index.append(agent_index[-1] + n_target_of_each_sample[idx])
    return {"batched_graph": dgl.batch(batched_graph),
            "batched_targets": th.cat(batched_target),
            "batched_track_mask": th.cat(batched_av_mask),
            "batched_agent_index": th.IntTensor(agent_index),
            "batched_transform_radix": np.stack(batched_transform_radix),
            "city": city,
            "seq_id": seq_id
            }


class AllDataset:
    def __init__(self,
                 train_dir='./train/forecasting_train_head_10000',
                 train_fraction=1.0,
                 val_dir='./train/forecasting_train_head_1000',
                 val_fraction=1.0,
                 test_dir='./train/forecasting_train_head_1000',
                 test_fraction=1.0,
                 ):
        self.train = MyDataset(raw_dir=train_dir, fraction=train_fraction)
        self.val = MyDataset(raw_dir=val_dir, fraction=val_fraction)
        self.test = MyDataset(raw_dir=test_dir, fraction=test_fraction)

if __name__ == "__main__":
    # dataset = MyDataset(raw_dir='/home/huanghao/Lab/argodataset/test_obs/data', fraction=1.00002)

    dataset = MyDataset(raw_dir='./train/forecasting_val_v1.1', fraction=1.015)
    # dataset = MyDataset(raw_dir='./train/forecasting_train_head_10000', fraction=1.002)
    # dataset = MyDataset(raw_dir='./test/test_sample', fraction=1.002)
    start_time = time.time()
    from dgl.dataloading import GraphDataLoader
    data_loader = GraphDataLoader(dataset, collate_fn=collate, batch_size=20, shuffle=False, drop_last=True)
    for _ in range(1):
        print(f"epoch : {_}-------------------------------------------------------------------")
        for i, gd in enumerate(data_loader):
            print(f"{i} | {time.time() - start_time:6.2f} s")
            g, t, m, a, r, c, s = gd["batched_graph"], \
                                    gd["batched_targets"], \
                                    gd["batched_track_mask"], \
                                     gd["batched_agent_index"], \
                                     gd["batched_transform_radix"], \
                                      gd["city"], \
                                        gd["seq_id"]
            print(f"id: {s}")
            # print(g)
            # print(f"observe{g.nodes['track_point'].data['state'].shape}")
            # print(g.nodes['track_point'].data['state'])
            # print(f"target features size:{t.shape}")
            # t_trans = t.view(-1, 30, 2)
            # print(t_trans)
            # print(t_trans.shape)
            # print(f"mask size:{m.shape}")
            # print(m, m.sum())
            # print(f"agent track size:{a.shape}")
            # print(a)
            # print(f"transform size:{r.shape}")
            # print(r)
