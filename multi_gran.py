import torch
from torch import nn
from d2l import torch as d2l


class Multi_gran(nn.Module):
    def __init__(self, series_dim, hidden_size, num_layers, num_grans):
        super(Multi_gran, self).__init__()
        # self.gru = nn.GRU
        self.input_size = self.output_size = series_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = [nn.LSTM(series_dim, hidden_size, num_layers) for _ in range(num_grans)]
        self.prediction_linear = [nn.Linear(hidden_size, series_dim) for _ in range(num_grans)]
        self.reconstruction_linear = [nn.Linear(hidden_size, series_dim) for _ in range(num_grans)]

    def __call__(self, inputs, state):
        loss_fn = nn.MSELoss()
        granular_predictions = []
        l = 0
        # 插个眼 l=0
        for granular_index, X in enumerate(inputs):
            # X是一个时间窗口的所有点
            if granular_index != 0:
                l += loss_fn(X, prev_reconstruction_values)
                current_gran_X = X - prev_reconstruction_values
            else:
                current_gran_X = X
            hiddens, last_state = self.lstm[granular_index](current_gran_X, state)
            prediction = self.prediction_linear(last_state)
            reconstruction_values = self.reconstruction_linear(hiddens)
            prev_reconstruction_values = reconstruction_values


    def init_hiddens(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True)
        h, c = h.to(device), c.to(device)
        return (h, c)
