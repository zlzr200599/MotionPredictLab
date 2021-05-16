import torch
import torch.nn as nn
# import torch.nn.functional as F
import dgl
import dgl.function as fn
# import time
from dgl.nn.pytorch import Sequential
from dgl.nn import GATConv
from typing import Dict
import torch.nn.functional as F
import time

# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            activation()
        )

    def forward(self, x):
        output = self.MLP(x)
        return output


def copy_c_to_f(graph: dgl.DGLGraph, edge_name: str, from_h: str, to_h: str):
    def udf_reduce(nodes):
        return {to_h: torch.cat(nodes.mailbox['m'])}
    graph[edge_name].update_all(fn.copy_src(from_h, 'm'), udf_reduce, etype=edge_name)

class TrackLayer(nn.Module):
    def __init__(self, input_size):
        super(TrackLayer, self).__init__()
        self.mlp = MLP(input_size=input_size, hidden_size=input_size)

    def forward(self, graph: dgl.DGLGraph, inputs):
        with graph.local_scope():
            graph.nodes['track_point'].data['state'] = inputs['track_point_feats']
            graph.multi_update_all(
                {
                    'point_to_track':
                        (
                            lambda edges:
                            {'msg': self.mlp(edges.src['state'])}, fn.max('msg', 'pooling')
                         ),
                },
                "sum"
            )
            graph.multi_update_all(
                {
                    'track_to_point': (fn.copy_src('pooling', 'msg'), fn.sum('msg', 'pooling')),
                },
                "sum"
            )
            out_features = torch.cat(
                tensors=(graph.nodes['track_point'].data['state'], graph.nodes['track_point'].data['pooling']),
                dim=1,
            )
            return {'track_point_feats': out_features, 'track_feats': graph.nodes['track_graph'].data['pooling']}


class LaneLayer(nn.Module):
    def __init__(self, input_size):
        super(LaneLayer, self).__init__()
        self.mlp = MLP(input_size=input_size, hidden_size=input_size)

    def forward(self, graph: dgl.DGLGraph, inputs):
        with graph.local_scope():
            graph.nodes['lane_point'].data['state'] = inputs['lane_point_feats']
            graph.multi_update_all(
                {
                    'point_to_lane':
                        (
                            lambda edges:
                            {'msg': self.mlp(edges.src['state'])}, fn.max('msg', 'pooling')
                         ),
                },
                "sum"
            )
            graph.multi_update_all(
                {
                    'lane_to_point': (fn.copy_src('pooling', 'msg'), fn.sum('msg', 'pooling')),
                },
                "sum"
            )
            out_features = torch.cat(
                tensors=(graph.nodes['lane_point'].data['state'], graph.nodes['lane_point'].data['pooling']),
                dim=1,
            )
            return {'lane_point_feats': out_features, 'lane_feats': graph.nodes['lane_graph'].data['pooling']}


class SubTrackModel(nn.Module):
    def __init__(self, input_size, num_layers, output_size):
        super(SubTrackModel, self).__init__()
        self.model = Sequential(*(
            TrackLayer(input_size=input_size*2**i)
            for i in range(num_layers)
        ))
        self.out_linear_model = nn.Linear(in_features=input_size*2**(num_layers-1),
                                          out_features=output_size,
                                          bias=True,
                                          )

    def forward(self, graph: dgl.DGLGraph, inputs):
        model_out = self.model(graph, inputs)
        return self.out_linear_model(model_out['track_feats'])


class SubLaneModel(nn.Module):
    def __init__(self, input_size, num_layers, output_size):
        super(SubLaneModel, self).__init__()
        self.model = Sequential(*(
            LaneLayer(input_size=input_size*2**i)
            for i in range(num_layers)
        ))
        self.out_linear_model = nn.Linear(in_features=input_size*2**(num_layers-1),
                                          out_features=output_size,
                                          bias=True,
                                          )

    def forward(self, graph: dgl.DGLGraph, inputs):
        model_out = self.model(graph, inputs)
        return self.out_linear_model(model_out['lane_feats'])


class TrackSubGraph(nn.Module):
    def __init__(self, ):
        super(TrackSubGraph, self).__init__()

    def forward(self, graph: dgl.DGLGraph, inputs):
        return


class FieldNet(nn.Module):
    def __init__(self,
                 d_in_track_feats=4,
                 d_in_lane_feats=2,
                 n_track_layers=1,
                 n_lane_layers=1,
                 d_in_att_feats=10,
                 d_out_att_feats=2
                 ):
        super(FieldNet, self).__init__()
        self.sub_track_model = SubTrackModel(
            input_size=d_in_track_feats,
            num_layers=n_track_layers,
            output_size=d_in_att_feats
        )
        self.sub_lane_model = SubLaneModel(
            input_size=d_in_lane_feats,
            num_layers=n_lane_layers,
            output_size=d_in_att_feats
        )
        self.gat_model = GATConv(
            in_feats=d_in_att_feats,
            out_feats=d_out_att_feats,
            num_heads=60//d_out_att_feats
        )
        # self.decoder = MLP(d_out_feats, 60)

    def forward(self, graph: dgl.DGLGraph, inputs: Dict):
        track_feats = self.sub_track_model(graph, inputs)

        if inputs.get('lane_point_feats', None) is not None:
            lane_feats = self.sub_lane_model(graph, inputs)
            feats = torch.cat((track_feats, lane_feats))
        else:
            feats = track_feats

        # n_nodes = feats.shape[0]
        # u = [i for i in range(n_nodes) for _ in range(n_nodes)]
        # v = [j for _ in range(n_nodes) for j in range(n_nodes)]
        # att_graph = dgl.graph((u, v))

        att_graph = dgl.edge_type_subgraph(graph=graph, etypes=['track_to_track', 'lane_to_lane',
                                                                'lane_to_track', 'track_to_lane'])
        att_graph = dgl.to_homogeneous(att_graph)
        att_output = self.gat_model(att_graph, feats)
        # print(att_graph)
        att_output_flatten = torch.flatten(att_output, start_dim=1, end_dim=2)
        return att_output_flatten.index_select(dim=0, index=torch.tensor(range(len(track_feats))))
        # return att_output_flatten

if __name__ == "__main__":
    g1 = dgl.heterograph({
        ('track_point', 'point_to_track', 'track_graph'): ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 1, 1, 1]),
        ('track_graph', 'track_to_point', 'track_point'): ([0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8]),

        ('lane_point', 'point_to_lane', 'lane_graph'): ([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 2, 2, 2, 3, 3, 3]),
        ('lane_graph', 'lane_to_point', 'lane_point'): ([2, 2, 2, 2, 2, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8]),

        ('track_graph', 'track_to_track', 'track_graph'): (
            list(range(2)) * 2,
            [i for i in range(2) for _ in range(2)]
        ),
        ('lane_graph', 'lane_to_lane', 'lane_graph'): (
            list(range(4)) * 4,
            [i for i in range(4) for _ in range(4)]
        ),
        ('track_graph', 'track_to_lane', 'lane_graph'): (
            list(range(2)) * 4,
            [i for i in range(4) for _ in range(2)]
        ),
        ('lane_graph', 'lane_to_track', 'track_graph'): (
            list(range(4)) * 2,
            [i for i in range(2) for _ in range(4)]
        ),
    })

    g2 = dgl.heterograph({
        ('track_point', 'point_to_track', 'track_graph'): ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0]),
        ('track_graph', 'track_to_point', 'track_point'): ([0, 0, 0, 0, 0], [0, 1, 2, 3, 4]),

        ('lane_point', 'point_to_lane', 'lane_graph'): ([0, 1, 2], [0, 0, 0]),
        ('lane_graph', 'lane_to_point', 'lane_point'): ([0, 0, 0], [0, 1, 2]),

        ('track_graph', 'track_to_track', 'track_graph'): (
            list(range(1)) * 1,
            [i for i in range(1) for _ in range(1)]
        ),
        ('lane_graph', 'lane_to_lane', 'lane_graph'): (
            list(range(1)) * 1,
            [i for i in range(1) for _ in range(1)]
        ),
        ('track_graph', 'track_to_lane', 'lane_graph'): (
            list(range(1)) * 1,
            [i for i in range(1) for _ in range(1)]
        ),
        ('lane_graph', 'lane_to_track', 'track_graph'): (
            list(range(1)) * 1,
            [i for i in range(1) for _ in range(1)]
        ),
    })

    g = dgl.batch([g1, g2])
    field_net = FieldNet(d_in_track_feats=4,
                         d_in_lane_feats=2,
                         n_track_layers=2,
                         n_lane_layers=2,
                         d_in_att_feats=10,
                         d_out_att_feats=2
                         )
    track_out = field_net(g,
                          {'track_point_feats': torch.rand((14, 4)),
                           'lane_point_feats': torch.rand((12, 2)),
                           'train_mask': torch.IntTensor([0, 1]),
                           },
                          )
    print(track_out)
    print(track_out.shape)
