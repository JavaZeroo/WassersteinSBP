from calendar import c
import numpy as np
import matplotlib.pyplot as plt
import ot
import torch
from utils.Models import MLP
from utils.Datasets import BBdataset

from torch import nn, optim
import torchvision
from torch.utils.data import Dataset, DataLoader 
from tqdm.auto import tqdm
from pathlib import Path
import seaborn as sns
# from utils import save_gif_frame

torch.manual_seed(233)
np.random.seed(233)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# set gpu 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

experiment_name = "gaussian2mnist"  # 你可以根据需要动态设置这个变量
# log_dir = Path('experiments') / experiment_name / 'test' / time.strftime("%Y-%m-%d/%H_%M_%S/")
log_dir = Path('experiments') / experiment_name
log_dir.mkdir(parents=True, exist_ok=True)

def calculate_n_linear_interpolation(marginals, num_timepoints):
    ret = []
    n = num_timepoints + 1
    for i in range(n):  # 0<=alpha<=1
        ret.append((i-1)/(n-1)*marginals[:,1] + (n-i)/(n-1)*marginals[:,0])
    return np.array(ret)

ti = 80
p = 2
x = np.linspace(1/ti, 1, num=ti)
cost = np.absolute(x - np.expand_dims(x, axis=1))
cost = np.power(cost, p)
cost /= cost.max()

SIGMA = 1 # 0.02
EPSILON = 0.05

mnist_ds = torchvision.datasets.MNIST(
    root="./data/", 
    train=True, 
    download=True
    )

filter_number_1 = 8
filter_number_2 = 3

tgt_imgs_1 = []
# tgt_imgs_2 = []
# status = []
for img, label in mnist_ds:
    tgt_imgs_1.append(torch.Tensor(np.array(img)))
    # if label == filter_number_1:
    # elif label == filter_number_2:
    #     tgt_imgs_2.append(torch.Tensor(np.array(img)))
print(len(tgt_imgs_1))
# print(len(tgt_imgs_2))
tgt_imgs_1 = torch.stack(tgt_imgs_1)
# tgt_imgs_2 = torch.stack(tgt_imgs_2)

# normalize
tgt_imgs_1 = ((tgt_imgs_1 - tgt_imgs_1.min()) / (tgt_imgs_1.max() - tgt_imgs_1.min())) * 2
# tgt_imgs_2 = ((tgt_imgs_2 - tgt_imgs_2.min()) / (tgt_imgs_2.max() - tgt_imgs_2.min())) * 2

all_samples = len(tgt_imgs_1)
n_samples = int(all_samples / 1000) * 1000
n_samples = 1000
indices = torch.randperm(all_samples)[:n_samples]

# left sample used for test
test_samples = all_samples - n_samples
test_samples = int(test_samples / 100) * 100

print("Original samples: ", len(tgt_imgs_1))
print("Filtered samples: ", n_samples)


train_tgt_imgs_1 =  tgt_imgs_1[indices].unsqueeze(1)
test_tgt_imgs_1 = tgt_imgs_1[-test_samples:].unsqueeze(1)

gauss_samples = torch.randn_like(train_tgt_imgs_1)


# 生成样本
# gauss_samples = torch.Tensor(generate_gaussian_samples(mean, cov, n_samples))
# P1_samples = torch.Tensor(generate_moons_samples(n_samples, noise))
# Pn_samples = torch.Tensor(generate_s_curve_samples(n_samples, noise))


dists = [
    gauss_samples, 
    # torch.mean(torch.stack([train_tgt_imgs_1, train_tgt_imgs_2]), dim=0), 
    train_tgt_imgs_1, 
    # train_tgt_imgs_2
    ]
train_pair_list = [(0, 1)]
fig,axs = plt.subplots(1, len(dists), figsize=(len(dists)*5, 5))
for i in range(len(dists)):
    axs[i].imshow(dists[i][0][0])
    axs[i].set_title(f'Samples (i={i+1})')
    axs[i].set_xlabel('X')
    axs[i].set_ylabel('Y')
fig.tight_layout()
fig.show()
fig.savefig(log_dir / 'samples.png')



# 生成二维Brownian bridge
def gen_bridge_2d(x, y, ts, T, num_samples):
    sigma = SIGMA
    bridge = torch.zeros((ts.shape[0], num_samples, 1 , 28, 28))
    drift = torch.zeros((ts.shape[0], num_samples, 1, 28, 28))
    bridge[0] = x
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]
        dydt = (y - bridge[i]) / (T - ts[i])
        drift[i, :] = dydt
        diffusion = sigma * torch.sqrt(dt) * torch.randn_like(dydt)
        bridge[i+1] = bridge[i] + dydt * dt
        bridge[i+1, :] += diffusion
    return bridge, drift

def gen_2d_data(source_dist, target_dist, epsilon=EPSILON, T=1):
    ts = torch.arange(0, T+epsilon, epsilon)
    source_dist = torch.Tensor(source_dist)
    target_dist = torch.Tensor(target_dist)
    assert source_dist.shape == target_dist.shape
    num_samples = len(source_dist)
    bridge, drift = gen_bridge_2d(source_dist, target_dist, ts, T=T, num_samples=num_samples)
    return ts, bridge, drift


def train(model, train_dl, optimizer, scheduler, loss_fn, epoch_iterator):
    losses = 0
    for training_data in train_dl:
        x = training_data['x'].to(device)
        y = training_data['y'].to(device)
        t = training_data['t'].to(device)
        direction = training_data['direction'].to(device)
        if 'status' in training_data:
            status = training_data['status'].to(device)
        else:
            status = None
        pred = model(x, t, direction, status)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        losses = loss.item()
        cur_lr = optimizer.param_groups[-1]['lr']
        epoch_iterator.set_description("Training (lr: %2.5f)  (loss=%2.5f)" % (cur_lr, losses))
        
    return losses


class BasicDataset(Dataset):
    def __init__(self, ts, bridge, drift, direction, status=None):
        # scaled_tensor = normalized_tensor * 2 - 1
        self.times = ts[:len(ts)-1].repeat(n_samples,)
        self.positions = torch.cat(torch.split(bridge[:-1, :], 1, dim=1), dim=0)[:, 0]
        self.scores = torch.cat(torch.split(drift[:-1, :], 1, dim=1), dim=0)[:,0]
        self.direction = torch.Tensor([direction])
        self.status = status
        # self.raw_data = torch.concat([positions, scores], dim=-1)
        
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        ret = {
            'x': self.positions[index], 
            'y': self.scores[index],
            't': self.times[index],
            'direction': self.direction,
            }

        if self.status is not None:
            ret['status'] = self.status[index]
        return ret
from torch.utils.data import ConcatDataset
from utils.unet import UNetModel
model_list = []
import gc
def get_dl(src_id, tgt_id, dists, batch_size):
    src_dist, tgt_dist = torch.Tensor(dists[src_id]),torch.Tensor(dists[tgt_id])
    gc.collect()
    
    print("Generate Forward Data")
    ts, bridge_f, drift_f = gen_2d_data(src_dist, tgt_dist, epsilon=EPSILON, T=1/2)
    print("Generate Backward Data")
    ts, bridge_b, drift_b = gen_2d_data(tgt_dist, src_dist, epsilon=EPSILON, T=1/2)

    print(ts.shape, bridge_f.shape, drift_f.shape)
    dataset1 = BasicDataset(ts, bridge_f, drift_f, 0)
    dataset2 = BasicDataset(ts, bridge_b, drift_b, 1)
    combined_dataset = ConcatDataset([dataset1, dataset2])

    train_dl = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    return train_dl

checkpoint_path = Path('/home/ljb/WassersteinSBP/experiments/gaussian2mnist')
# checkpoint_path = None
# continue_train = True
continue_train = False
for index, pair in enumerate(train_pair_list):
    src_id, tgt_id = pair

    epochs = 100
    batch_size = 512
    lr = 1e-3


    image_size=28
    image_channels=1
    
    num_channels = 32
    num_res_blocks = 4
    num_heads = 4
    num_heads_upsample = -1
    attention_resolutions = "8"
    dropout = 0.1
    use_checkpoint = False
    use_scale_shift_norm = True

    channel_mult = (1, 2, 2)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    kwargs = {
        "in_channels": image_channels,
        "model_channels": num_channels,
        "out_channels": image_channels,
        "num_res_blocks": num_res_blocks,
        "attention_resolutions": tuple(attention_ds),
        "dropout": dropout,
        "channel_mult": channel_mult,
        "num_classes": None,
        "use_checkpoint": use_checkpoint,
        "num_heads": num_heads,
        "num_heads_upsample": num_heads_upsample,
        "use_scale_shift_norm": use_scale_shift_norm,
    }

    model = UNetModel(**kwargs).to(device)
    # 组合成data
    # train_ds = BBdataset(raw_data[:,0])
    refresh_rate = 3
    # for k,v in batch.items():
    #     print(k, v.shape)
    # 3 128
    # model = MLP(input_dim=4, output_dim=2, hidden_layers=4, hidden_dim=256, act=nn.LeakyReLU()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    loss_fn = nn.MSELoss()
    loss_list = []
    print('='*10+'model'+'='*10)
    # print(model)
    print('='*10+'====='+'='*10)
    if checkpoint_path is not None and not continue_train:
        laod_checkpoint_from = checkpoint_path / f"model_{index}.pt"
        print(f'Load Checkpoint from {laod_checkpoint_from}')
        model.load_state_dict(torch.load(laod_checkpoint_from))
    else:
        if checkpoint_path is not None and continue_train:
            laod_checkpoint_from = checkpoint_path / f"model_{index}.pt"
            print(f'Load Checkpoint from {laod_checkpoint_from}')
            model.load_state_dict(torch.load(laod_checkpoint_from))
        epoch_iterator = tqdm(range(epochs), desc="Training (lr: X)  (loss= X)", dynamic_ncols=True)
        model.train()
        for e in epoch_iterator:
            if e == 0 or (refresh_rate != 0 and e%refresh_rate==0):
                train_dl = get_dl(src_id, tgt_id, dists, batch_size)
            now_loss = train(model ,train_dl, optimizer, scheduler, loss_fn, epoch_iterator)
            loss_list.append(now_loss)
            cur_lr = optimizer.param_groups[-1]['lr']
            epoch_iterator.set_description("Training (lr: %2.5f)  (loss=%2.5f)" % (cur_lr, now_loss))
        plt.figure()
        plt.plot(loss_list)
        plt.savefig(log_dir / f'loss_{src_id}_{tgt_id}.png')
        plt.gca().cla()
        epoch_iterator.close()
        torch.save(model.state_dict(), log_dir / f"model_{index}.pt")
    model_list.append(model)
plt.show()

def inference(model, test_ts, test_source_sample, test_num_samples, reverse=False):
    model.eval()
    model.cpu()
    test_ts = test_ts[:-1]
    sigma = SIGMA
    pred_bridge = torch.zeros(len(test_ts), test_num_samples, 1, 28, 28)
    pred_drift = torch.zeros(len(test_ts)-1, test_num_samples, 1, 28, 28)
    pred_bridge[0, :] = test_source_sample
    with torch.no_grad():
        for i in tqdm(range(len(test_ts) - 1)):
            dt = abs(test_ts[i+1] - test_ts[i])
            if reverse:
                direction = torch.ones_like(test_ts[i:i+1])
            else:
                direction: torch.Tensor = torch.zeros_like(test_ts[i:i+1])
            dydt = model(pred_bridge[i], test_ts[i:i+1], direction, None)
            diffusion = sigma * torch.sqrt(dt) * torch.randn(test_num_samples, 1, 28, 28)
            pred_drift[i, :] = dydt
            pred_bridge[i+1, :] = pred_bridge[i, :] + dydt * dt
            pred_bridge[i+1, :] += diffusion
    return pred_bridge, pred_drift


# 生成样本
test_num_samples = 25
test_P2_samples = test_tgt_imgs_1[25:test_num_samples+25]
# test_P3_samples = test_tgt_imgs_2[:test_num_samples]
test_P1_samples = torch.randn_like(test_P2_samples)
test_ts, test_bridge, test_drift = gen_2d_data(test_P1_samples, test_P2_samples, epsilon=0.01, T=1)

test_pred_bridges = []
test_pred_drifts = []
infer_chain = [
    (0,)
    # (0,2),
    # (0,1), 
    # (-1,2),
    # (-2,1)
    ]
# for chain in infer_chain:
#     chain_out = []
#     drifts = []
#     if chain[0] == 0:
#         temp_src = torch.Tensor(test_P1_samples)  
#     elif chain[0] == -1:
#         temp_src = torch.Tensor(test_P2_samples)
#     elif chain[0] == -2:
#         temp_src = torch.Tensor(test_P3_samples)
        
        
#     for i in chain:
#         print(i)
#         model = model_list[abs(i)]
#         pred_bridge, pred_drift = inference(model, test_ts[:len(test_ts)//len(chain)], temp_src, test_num_samples, reverse=i<0)
#         chain_out.append(pred_bridge)
#         drifts.append(pred_bridge)
#         temp_src = chain_out[-1][-1, :, :].clone()
#     test_pred_bridges.append(chain_out)
#     test_pred_drifts.append(drifts)


def draw_comapre_split(test_pred_bridges):
    n_sub_interval = len(test_pred_bridges)+1
    print(n_sub_interval)
    fig, axs = plt.subplots(1, n_sub_interval, figsize=(5*n_sub_interval, 5))
    print(axs.shape)
    def plot_test_pred_bridges(sub_axs, data):
        for i in range(n_sub_interval):
            now = data[i][0, :] if i != n_sub_interval-1 else data[i-1][-1, :]
            combined_image = torch.cat([torch.cat([now[j] for j in range(k, k+5)], dim=2) for k in range(0, 25, 5)], dim=1)
            combined_image = (combined_image-combined_image.min())/(combined_image.max()-combined_image.min())
            # combined_image[combined_image<0.3] = 0
            # combined_image[combined_image>0.7] = 1
            
            # combined_image = (combined_image-combined_image.min())/(combined_image.max()-combined_image.min())
            sub_axs[i].imshow(combined_image.permute(1,2,0).numpy(), cmap='gray')
            
    plot_test_pred_bridges(axs, test_pred_bridges)

    # set tight layout
    fig.tight_layout()
    
    # fig
    fig.show()
    
    return fig

def draw_comapre(dists, test_pred_bridges, test_pred_bridges2, test_pred_bridges3, test_pred_bridges4, bound=12):
    n_sub_interval = len(dists)-1
    fig, axs = plt.subplots(4, n_sub_interval, figsize=(5*n_sub_interval, 20))

    def plot_test_pred_bridges(sub_axs, data):
        for i in range(n_sub_interval):
            now = data[i][0, :] if i != n_sub_interval-1 else data[i-1][-1, :]
            combined_image = torch.cat([torch.cat([now[j, 0] for j in range(k, k+5)], dim=1) for k in range(0, 25, 5)], dim=0)
            sub_axs[i].imshow(combined_image, cmap='gray')
            
    plot_test_pred_bridges(axs[0], test_pred_bridges)
    axs[0][0].set_ylabel('Chain 0 -> 1 -> 2')
    
    axs[1][0].set_ylabel('Chain 0 -> 1 -> 3')
        
    
    plot_test_pred_bridges(axs[1], test_pred_bridges2)

    plot_test_pred_bridges(axs[2], test_pred_bridges3)
    axs[2][0].set_ylabel('Chain 2 -> 1 -> 3')
    plot_test_pred_bridges(axs[3], test_pred_bridges4)
    axs[3][0].set_ylabel('Chain 3 -> 1 -> 2')

    # set tight layout
    fig.tight_layout()
    
    # fig
    fig.show()
    
    return fig
    
# draw_comapre(dists, test_pred_bridges[0], test_pred_bridges[1], test_pred_bridges[2], test_pred_bridges[3]).savefig(log_dir / 'compare.png')

# for i in range(len(test_pred_bridges)):
#     draw_comapre_split(test_pred_bridges[i]).savefig(log_dir / f'compare_{i}.png')

import imageio
import shutil
from rich.progress import track

def save_gif_frame(bridge, save_path=None, name='brownian_bridge.gif', bound=10):
    assert save_path is not None, "save_path cannot be None"
    save_path = Path(save_path)
    # bridge = bridge[:, :, :].numpy()  # 降低采样率

    temp_dir = save_path / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    frame = 0
    
    for i in track(range(bridge.shape[0]), description="Processing image"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        now = bridge[i, :]
        combined_image = torch.cat([torch.cat([now[j, 0] for j in range(k, k+5)], dim=1) for k in range(0, 25, 5)], dim=0)

        ax.imshow(combined_image.numpy(), cmap='gray')
        fig.savefig(save_path / 'temp' / f'{frame:03d}.png', dpi=100)
        frame += 1
        fig.show()
        plt.close('all')
    frames = []
    for i in range(bridge.shape[0]):
        frame_image = imageio.imread(save_path / 'temp' / f'{i:03d}.png')
        frames.append(frame_image)
    imageio.mimsave(save_path / name, frames, duration=0.2)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

save_gif_frame(test_bridge, log_dir, name="pred_0.gif", bound=15)

# for i, test_pred_bridge in enumerate(test_pred_bridges):
#     save_gif_frame(torch.concat(test_pred_bridge, dim=0), log_dir, name=f"pred_{i}.gif", bound=15)