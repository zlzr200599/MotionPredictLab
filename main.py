from data.argodataset import AllDataset, collate
import time
import torch
from dgl.dataloading import GraphDataLoader
# from torch.utils.data import DataLoader
from models.FieldNet import FieldNet
import warnings
from collections import deque
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data = AllDataset(train_dir='./data/train/forecasting_train_head_10000',
                      train_fraction=1.0,
                      val_dir='./data/train/forecasting_val_v1.1',
                      val_fraction=1.0,
                      test_dir='./data/train/forecasting_val_v1.1',
                      test_fraction=1.0,
                      )

    data_loader = GraphDataLoader(data.train, collate_fn=collate, batch_size=10, shuffle=True, drop_last=True)
    model = FieldNet(
        d_in_track_feats=4,
        d_in_lane_feats=2,
        n_track_layers=1,
        n_lane_layers=1,
        d_in_att_feats=10,
        d_out_att_feats=2
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00000)
    start_time = time.time()
    loss_queue = deque(maxlen=20)
    real_queue = deque(maxlen=20)
    for e in range(10):
        print(f"epoch : {e}----------------------------------------------------------------")
        for i, gd in enumerate(data_loader):
            # print(f"epoch: {e} | n_iter: {i} | {time.time() - start_time:6.2f} s ")
            g, t, m, a, r, c, s = gd["batched_graph"], \
                                  gd["batched_targets"], \
                                  gd["batched_track_mask"], \
                                  gd["batched_agent_index"], \
                                  gd["batched_transform_radix"], \
                                  gd["city"], \
                                  gd["seq_id"]

            y_pred = model(g,
                           {'track_point_feats': g.nodes['track_point'].data['state'],
                            'lane_point_feats': g.nodes['lane_point'].data['state'],
                            },
                           )
            y_pred = y_pred * m
            y_pred = y_pred.index_select(0, a)
            t = t.index_select(0, a)
            loss = criterion(y_pred, t)
            real_lose = torch.square(y_pred - t).flatten().view(-1, 2)
            real_lose = torch.sum(real_lose, dim=1)
            real_lose = torch.sqrt(real_lose)
            real_lose = torch.mean(real_lose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_queue.append(loss)
            real_queue.append(real_lose)
            #         assert False
            if i % 10 == 0:
                print(
                    f"epoch: {e} | n_iter: {i} |loss : {sum(loss_queue) / len(loss_queue):6.4f} | {time.time() - start_time:6.2f} s ")
                print(f"real loss average error: {sum(real_queue) / len(real_queue):6.4f} m")
