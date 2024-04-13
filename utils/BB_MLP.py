import random
import torch
import torch.nn as nn

class Brownian_MLP(nn.Module):
    def __init__(self, input_channel=2, output_channel=1, hidden_layer=5, dropout=0.1, bias=False):
        super(Brownian_MLP, self).__init__()
        self.input_shape = input_channel
        self.output_shape = output_channel
        self.bias = bias
        self.hidden_layer = hidden_layer
        self.droput = dropout
        self.network = nn.Sequential()
        self.criterion = nn.MSELoss()

        hid_dims = [random.randint(64, 96) for _ in range(hidden_layer + 1)]
        self.network.add_module('Input_Layer', nn.LazyLinear(hid_dims[0]))
        self.network.add_module('ReLU', nn.ReLU())
        for i in range(hidden_layer):
            self.network.add_module(f'Hidden_{i}', nn.LazyLinear(hid_dims[i+1], bias=self.bias))
            self.network.add_module(f'Tanh_{i}', nn.Tanh())
            self.network.add_module(f'Droput_{i}', nn.Dropout(self.droput))

        self.network.add_module('Output_Layer', nn.LazyLinear(output_channel))


    def forward(self, input):
        '''
            input structure: a data dict = {
                'source': a tensor size=(batch, channel),
                't': a number represent the time
            }

            result: a tensor size=(batch, channel)
        '''
        device = input['source'].device

        src = input['source'].unsqueeze(-1)
        t = input['t'] * torch.ones(src.size()).to(device)
        x = torch.concat([src, t], dim=-1)
        result = self.network(x)
        return result
    
    def cal_loss(self, result, target):
        loss = self.criterion(result, target)
        return loss
    
class Brownian_Model(nn.Module):
    def __init__(self, embedding_size=16, output_channel=1, hidden_layer=5, dropout=0.1, bias=False):
        super(Brownian_Model, self).__init__()
        self.embedding_size = embedding_size
        self.output_shape = output_channel
        self.bias = bias
        self.hidden_layer = hidden_layer
        self.droput = dropout
        self.network = nn.Sequential()
        self.criterion = nn.MSELoss()

        # hid_dims = [random.randint(64, 96) for _ in range(hidden_layer + 1)]
        hid_dims = [128 for _ in range(hidden_layer + 1)]
        self.network.add_module('Input_Layer', nn.LazyLinear(hid_dims[0]))
        self.network.add_module('Tanh', nn.Tanh())
        for i in range(hidden_layer):
            self.network.add_module(f'Hidden_{i}', nn.LazyLinear(hid_dims[i+1], bias=self.bias))
            self.network.add_module(f'LeakyReLU_{i}', nn.LeakyReLU())
            # self.network.add_module(f'Droput_{i}', nn.Dropout(self.droput))

        self.network.add_module('Output_Layer', nn.LazyLinear(output_channel))

        self.time_encoder = TimeEncoder(embedding_size)
        self.position_encoder = PositionEncoder(embedding_size)

    def forward(self, input):
        '''
            input structure: a data dict = {
                'source': a tensor size=(batch, channel),
                't': a number represent the time
            }

            result: a tensor size=(batch, channel)
        '''
        device = input['source'].device
        src = input['source']

        t = input['t'] * torch.ones((len(src), 1)).to(device)
        t = self.time_encoder(t)

        p = self.position_encoder(src)

        x = torch.concat([p, t], dim=-1)
        result = self.network(x)
        return result
    
    def cal_loss(self, result, target):
        loss = self.criterion(result, target)
        return loss

class TimeEncoder(nn.Module):
    def __init__(self, output_channel=16, hidden_layer=1, dropout=0.1, bias=False):
        super(TimeEncoder, self).__init__()
        self.bias=bias
        self.dropout=dropout
        self.network = nn.Sequential()
        hid_dims = [64 for _ in range(hidden_layer + 1)]
        self.network.add_module('Input_Layer', nn.LazyLinear(hid_dims[0]))
        self.network.add_module('Tanh', nn.Tanh())
        for i in range(hidden_layer):
            self.network.add_module(f'Hidden_{i}', nn.LazyLinear(hid_dims[i+1], bias=self.bias))
            self.network.add_module(f'LeakyReLU_{i}', nn.LeakyReLU())
            # self.network.add_module(f'Droput_{i}', nn.Dropout(self.dropout))
        self.network.add_module('Output_Layer', nn.LazyLinear(output_channel))

    def forward(self, input):
        return self.network(input)
    
class PositionEncoder(nn.Module):
    def __init__(self, output_channel=16, hidden_layer=1, dropout=0.1, bias=False):
        super(PositionEncoder, self).__init__()
        self.bias=bias
        self.dropout=dropout
        self.network = nn.Sequential()
        hid_dims = [64 for _ in range(hidden_layer + 1)]
        self.network.add_module('Input_Layer', nn.LazyLinear(hid_dims[0]))
        self.network.add_module('Tanh', nn.Tanh())
        for i in range(hidden_layer):
            self.network.add_module(f'Hidden_{i}', nn.LazyLinear(hid_dims[i+1], bias=self.bias))
            self.network.add_module(f'LeakyReLU_{i}', nn.LeakyReLU())
            # self.network.add_module(f'Droput_{i}', nn.Dropout(self.dropout))
        self.network.add_module('Output_Layer', nn.LazyLinear(output_channel))

    def forward(self, input):
        return self.network(input)

class DimensionEncoder(nn.Module):
    def __init__(self, output_channel=5, hidden_layer=1, dropout=0.1, bias=False):
        super(DimensionEncoder, self).__init__()
        self.bias=bias
        self.dropout=dropout
        self.network = nn.Sequential()
        hid_dims = [random.randint(64, 96) for _ in range(hidden_layer + 1)]
        self.network.add_module('Input_Layer', nn.LazyLinear(hid_dims[0]))
        self.network.add_module('ReLU', nn.ReLU())
        for i in range(hidden_layer):
            self.network.add_module(f'Hidden_{i}', nn.LazyLinear(hid_dims[i+1], bias=self.bias))
            self.network.add_module(f'Tanh_{i}', nn.Tanh())
            self.network.add_module(f'Droput_{i}', nn.Dropout(self.dropout))
        self.network.add_module('Output_Layer', nn.LazyLinear(output_channel))

    def forward(self, input):
        return self.network(input)