from torch import nn
import torch
import math
import segmentation_models_pytorch as smp



class timeUnetPlusPlus(smp.UnetPlusPlus):
    def __init__(self, model_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # self.timeEmb
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.lns = nn.ModuleList()
        self.lns.append(nn.Linear(time_embed_dim, 2))
        # self.lns.append(nn.Linear(time_embed_dim, 32))
        # self.lns.append(nn.Linear(time_embed_dim, 24))
        # self.lns.append(nn.Linear(time_embed_dim, 40))
        self.lns.append(nn.Linear(time_embed_dim, 48))
        self.lns.append(nn.Linear(time_embed_dim, 32))
        self.lns.append(nn.Linear(time_embed_dim, 56))
        
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(2 * 2, 2, 1))
        # self.convs.append(nn.Conv2d(32 * 2, 32, 1))
        # self.convs.append(nn.Conv2d(24 * 2, 24, 1))
        # self.convs.append(nn.Conv2d(40 * 2, 40, 1))
        self.convs.append(nn.Conv2d(48 * 2, 48, 1))
        self.convs.append(nn.Conv2d(32 * 2, 32, 1))
        self.convs.append(nn.Conv2d(56 * 2, 56, 1))
        
        self.name = 'timeUnetPlusPlus'
        self.initialize()

    def timestep_embedding(self, timesteps, dim, max_period=1000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, timesteps):        
        emb = self.timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(emb)
        self.check_input_shape(x)
        features = self.encoder(x)
        for index, f in enumerate(features):
            temp_emb = self.lns[index](emb)
            while len(temp_emb.shape) < len(f.shape):
                temp_emb = temp_emb[..., None]
            # f = f + temp_emb
            f = torch.concat([f, temp_emb.repeat(1, 1, *f.shape[-2:])], dim=1)
            f = self.convs[index](f)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_dim=64, act: nn.Module=nn.ReLU()):
        super(MLP, self).__init__()
        
        self.fcin = nn.Linear(input_dim, hidden_dim)
        self.fcs = nn.ModuleList()
        for i in range(hidden_layers):
            self.fcs.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.BatchNorm1d(hidden_dim)))
        self.fcout = nn.Linear(hidden_dim, output_dim)
        self.act = act
        
    def forward(self, x):
        ret = self.fcin(x)
        ret = self.act(ret)
        for fc in self.fcs:
            ret = fc(ret)
            ret = self.act(ret)
        ret = self.fcout(ret)
        return ret
    
def get_model(args):
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    model = MLP(input_dim=args.input_dim, output_dim=args.output_dim, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim).to(args.device)

    # 如果有多个 GPU，则使用 DataParallel
    if not args.single_gpu and  torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(args.device)

    return model
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer_dim, dropout_prob=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # 添加额外的全连接层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out, hidden
    


if __name__ == "__main__":
    model = LSTMModel(5, 2, 64, 16)
    x = torch.randn(100, 1, 5)
    hidden = None
    output, hidden = model(x, hidden)
    print(output.shape, hidden[0].shape, hidden[1].shape)
    output, hidden = model(x, hidden)
    print(output.shape, hidden[0].shape, hidden[1].shape)