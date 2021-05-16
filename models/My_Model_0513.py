import torch
import dgl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader
import dgl.function as fn
import torch.nn as nn
from data.mydataset import AllDataset, collate, MyDataset
import time
from collections import deque
import numpy as np
import pandas as pd
import os
from dgl.nn import GATConv
from torch.optim.lr_scheduler import LambdaLR
from utils import val_plot


class MyModel(nn.Module):
    def __init__(self, saved_path: str):
        super(MyModel, self).__init__()
        self.saved_path = saved_path
        self.training = False

        self.agent_encoder = nn.LSTM(input_size=2, hidden_size=25, num_layers=1, )
        self.agent_linear = nn.Linear(25, 2)
        self.agent_decoder = nn.LSTM(input_size=1, hidden_size=25, num_layers=1, )

        self.lane_encoder = nn.LSTM(input_size=2, hidden_size=25, num_layers=1, )

        self.av_encoder = nn.LSTM(input_size=2, hidden_size=25, num_layers=1, )
        self.others_encoder = nn.LSTM(input_size=2, hidden_size=25, num_layers=1, )

        self.conv = dglnn.HeteroGraphConv({'av_env': GATConv(in_feats=25, out_feats=5, num_heads=5,),
                                           'others_env': GATConv(in_feats=25, out_feats=5, num_heads=5,),
                                           'lane_env': GATConv(in_feats=25, out_feats=5, num_heads=5,)},
                                          aggregate='mean'
                                          )



    def forward(self, g: dgl.DGLGraph):
        #  INIT ----------------------------------------------------------------#
        agent: torch.FloatTensor = g.nodes['agent'].data['state'][:, :20, :]
        batch_size = agent.shape[0]
        av: torch.FloatTensor = g.nodes['av'].data['state']
        others: torch.FloatTensor = g.nodes['others'].data['state']
        lane: torch.FloatTensor = g.nodes['lane'].data['state']

        # ENCODER---------------------------------------------------------------#
        #  agent
        agent = torch.transpose(agent, dim0=0, dim1=1)
        agent_code, (agent_h_n, agent_c_n) = self.agent_encoder(agent)
        g.nodes['agent'].data['att'] = agent_c_n.squeeze(dim=0)
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

        # GRAPH-----------------------------------------------------------------#
        g.update_all(message_func=fn.copy_u('att', 'msg'),
                     reduce_func=fn.mean('msg', 'env_agent_info'),
                     etype='agent_env')
        g.update_all(message_func=fn.copy_u('code', 'msg'),
                     reduce_func=fn.mean('msg', 'env_lane_info'),
                     etype='lane_env')
        inputs = {'av': g.nodes['av'].data['code'],
                  'others': g.nodes['others'].data['code'],
                  'lane': g.nodes['lane'].data['code'],
                  'env': g.nodes['env'].data['env_agent_info'],
        }
        env_code = self.conv(g, inputs)['env']
        env_code = env_code.view(batch_size, -1).unsqueeze(0)
        env_feat: torch.FloatTensor = g.nodes['env'].data['env_lane_info'].unsqueeze(0)
        # print(g.nodes['env'].data['lane_env'].shape)
        # assert False, "输出卷积结果,batch_size = 16"

        # DECODER --------------------------------------------------------------#
        t = torch.arange(0.0, 3.0, 0.1).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, -1).permute(1, 0, 2)
        agent_decoder_h_0 = agent_h_n
        agent_decoder_c_0 = agent_c_n + env_code
        predict_agent_track, (h, c) = self.agent_decoder(input=t,
                                                         hx=(agent_decoder_h_0, agent_decoder_c_0)
                                                         )
        g.nodes['agent'].data['predict'] = self.agent_linear(predict_agent_track).permute(1, 0, 2)

    def train_model(self, dataset: AllDataset, collate_fn=collate, batch_size=10, shuffle=True, drop_last=True,
                    n_epoch=10, lr=0.05,
                    ):
        self.training = True
        data_loader = GraphDataLoader(dataset.train, collate_fn=collate_fn, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=drop_last)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 0.9 ** e, verbose=True)
        start_time = time.time()
        loss_queue = deque(maxlen=10)
        real_queue = deque(maxlen=10)
        for epoch in range(n_epoch):
            for i, (bhg, info) in enumerate(data_loader):
                optimizer.zero_grad()
                self.forward(bhg)
                y_pred = bhg.nodes['agent'].data['predict']
                y_true = bhg.nodes['agent'].data['state'][:, 20:, :]
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()

                loss_queue.append(loss)
                real_lose = torch.square(y_pred - y_true).flatten().view(-1, 2)
                real_lose = torch.sum(real_lose, dim=1)
                real_lose = torch.sqrt(real_lose)
                real_lose = torch.mean(real_lose)
                real_queue.append(real_lose)
                if i % 10 == 0:
                    print(
                        f"epoch: ({epoch}/{n_epoch}) | n_iter: ({i}/{len(data_loader)}) | "
                        f"loss : {sum(loss_queue) / len(loss_queue):6.4f} | "
                        f"{time.time() - start_time:6.2f} s "
                    )
                    print(f"real loss average error: {sum(real_queue) / len(real_queue):6.4f} m")
            scheduler.step()
            self.val_model(dataset=dataset)
            print("-------------------------------------------------------------------------------------------")
            print(f"have train: {epoch} epoch \n"
                  f"estimate time remain: {(n_epoch - epoch) * (time.time() - start_time)/(epoch+1):8.2f} s")
            print("-------------------------------------------------------------------------------------------")
            print(f"total time: {time.time() - start_time:6.2f} s")
        self.save()
        self.training = False

    @torch.no_grad()
    def val_model(self, dataset: AllDataset, return_to_plot=False):
        if not self.training:
            self.load()
        self.eval()
        data_loader = GraphDataLoader(dataset.val, collate_fn=collate,
                                      batch_size=int(10 if not return_to_plot else 1),
                                      shuffle=False, drop_last=False)
        start_time = time.time()
        real_queue = deque()
        for i, (bhg, info) in enumerate(data_loader):
            self.forward(bhg)
            y_pred = bhg.nodes['agent'].data['predict']
            y_true = bhg.nodes['agent'].data['state'][:, 20:, :]

            real_lose = torch.square(y_pred - y_true).flatten().view(-1, 2)
            real_lose = torch.sum(real_lose, dim=1)
            real_lose = torch.sqrt(real_lose)
            real_lose = torch.mean(real_lose)
            real_queue.append(real_lose)
            if return_to_plot:
                val_plot(bhg)
        print("-------------------------------------evaluation---------------------------------------------")
        print(f"val total time elapse: {time.time() - start_time:6.2f} s| #samples : {len(dataset.val)}"
              f" loss : {sum(real_queue) / len(real_queue):6.4f} m")
        print("--------------------------------------------------------------------------------------------")
        self.train()

    @torch.no_grad()
    def test_model(self, dataset: AllDataset, output_dir: str = "test_result"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"make new dir {os.path.abspath(output_dir)}, and write files into it.")
        else:
            print(f'output dir {os.path.abspath(output_dir)} exists !')
        self.load()
        self.eval()
        data_loader = GraphDataLoader(dataset.test, collate_fn=collate, batch_size=10,
                                      shuffle=False, drop_last=False)
        start_time = time.time()
        for i, (bhg, info) in enumerate(data_loader):
            batch_size = len(info)
            self.forward(bhg)
            y_pred: torch.FloatTensor = bhg.nodes['agent'].data['predict']
            assert batch_size == y_pred.shape[0]
            for n, d in enumerate(info):
                st = float(d['split_time'])
                x, y = d['radix']['x'], d['radix']['y']
                timestamp = pd.Series(np.linspace(st + 0.1, st + 3.0, 30, dtype=np.float), name="TIMESTAMP")
                track_id = pd.Series([d['agent_track_id'] for _ in range(30)], name="TRACK_ID")
                object_type = pd.Series(["AGENT" for _ in range(30)], name="OBJECT_TYPE")
                x = pd.Series(y_pred[n, :, 0] + x, name="X")
                y = pd.Series(y_pred[n, :, 1] + y, name="Y")
                city_name = pd.Series([d['city'] for _ in range(30)], name="CITY_NAME")
                this_df = pd.DataFrame(list(zip(timestamp, track_id, object_type, x, y, city_name)),
                                       columns=("TIMESTAMP", "TRACK_ID", "OBJECT_TYPE", "X", "Y", "CITY_NAME")
                                       )
                stack_df = pd.concat(objs=[d['df'], this_df])

                stack_df.to_csv(os.path.join(output_dir, d['filename']+".csv"), index=False)

                # pd.set_option('display.max_columns', 1000)
                # print(this_df)
        self.train()
        print(f"test time is :{time.time() - start_time:6.2f} s | num_samples : {len(dataset.test)}")


    def save(self):
        torch.save(self.state_dict(), self.saved_path)
        print(f'save the model to {os.path.abspath(self.saved_path)}')
        return self.saved_path

    def load(self):
        self.load_state_dict(torch.load(self.saved_path))


if __name__ == "__main__":
    n_av, n_other, n_lane = 1, 3, 5
    graph = dgl.heterograph({
        ('agent', 'agent_env', 'env'): ([0], [0]),
        ('av', 'av_env', 'env'): (list(range(n_av)), [0] * n_av),
        ('others', 'others_env', 'env'): (list(range(n_other)), [0] * n_other),
        ('lane', 'lane_env', 'env'): (list(range(n_lane)), [0] * n_lane),
    })

    graph.nodes['agent'].data['state'] = torch.randn((20, 2), dtype=torch.float).unsqueeze(dim=0)
    graph.nodes['av'].data['state'] = torch.randn((n_av, 20, 2), dtype=torch.float)
    # graph.nodes['av'].data['mask'] = th.tensor(av_tracks_mask, dtype=th.float)
    graph.nodes['others'].data['state'] = torch.rand((n_other, 20, 2), dtype=torch.float)
    # graph.nodes['others'].data['mask'] = th.tensor(other_tracks_mask, dtype=th.float)
    graph.nodes['lane'].data['state'] = torch.rand((n_lane, 20, 2), dtype=torch.float)
    model = MyModel()
    model(graph)
    print(graph.nodes['agent'].data['predict'])
