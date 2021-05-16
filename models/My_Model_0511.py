import torch
import dgl
from dgl.dataloading import GraphDataLoader
import dgl.function as fn
import torch.nn as nn
from data.mydataset import AllDataset, collate, MyDataset
import time
from collections import deque
import numpy as np
import pandas as pd
import os


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.agent_encoder = nn.LSTM(input_size=2, hidden_size=25, num_layers=1, )
        self.agent_decoder = nn.LSTM(input_size=1, hidden_size=25, num_layers=1, )
        self.agent_linear = nn.Linear(25, 2)
        # self.av_lstm = nn.LSTM(2, 20, 1, batch_first=False)
        # self.others_lstm = nn.LSTM(2, 20, 1, batch_first=False)
        # self.lane_lstm = nn.LSTM(2, 20, 1, batch_first=False)

    def forward(self, g: dgl.DGLGraph):
        agent = g.nodes['agent'].data['state'][:, :20, :]
        batch_size = agent.shape[0]
        # av = g.nodes['av'].data['state']
        # others = g.nodes['others'].data['state']
        # lane = g.nodes['lane'].data['state']
        agent = torch.transpose(agent, dim0=0, dim1=1)
        agent_out, (h_n, c_n) = self.agent_encoder(agent)
        # print(agent_out.shape, h_n.shape, c_n.shape)
        # av = self.agent_lstm(av)
        # others = self.others_lstm(others)
        # lane = self.lane_lstm(lane)
        t = torch.arange(0.0, 3.0, 0.1).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, -1).permute(1, 0, 2)
        predict_agent_track, (h, c) = self.agent_decoder(input=t, hx=(h_n, c_n))
        g.nodes['agent'].data['predict'] = self.agent_linear(predict_agent_track).permute(1, 0, 2)

    def train_model(self, dataset: AllDataset, collate_fn=collate, batch_size=10, shuffle=True, drop_last=True,
                    n_epoch=10, lr=0.05,
                    ):
        data_loader = GraphDataLoader(dataset.train, collate_fn=collate_fn, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=drop_last)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
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
            # model.save()
            self.val_model(dataset=dataset)
            print("-------------------------------------------------------------------------------------------")
            print(f"have train: {epoch} epoch \n"
                  f"estimate time remain: {(n_epoch - epoch) * (time.time() - start_time)/(epoch+1):8.2f} s")
            print("-------------------------------------------------------------------------------------------")
            print(f"total time: {time.time() - start_time:6.2f} s")
        self.save('new.pth')

    @torch.no_grad()
    def val_model(self, dataset: AllDataset, ):
        self.eval()
        data_loader = GraphDataLoader(dataset.val, collate_fn=collate, batch_size=10,
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

                this_df.to_csv(os.path.join(output_dir, d['filename']+".csv"), index=False)
                # pd.set_option('display.max_columns', 1000)
                # print(this_df)
        self.train()
        print(f"test time is :{time.time() - start_time:6.2f} s | num_samples : {len(dataset.test)}")

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + 'MyNet' + '_'
            name = time.strftime(prefix + '%Y%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))


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
