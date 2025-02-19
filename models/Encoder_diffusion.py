import torch
import torch.nn as nn
import numpy as np
import random
import os
import math
import logging
from tqdm import tqdm


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0")
torch.cuda.set_device(0)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.002, img_size=384, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x



class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        self.act = nn.SiLU()  # Swish activation
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, time_emb_channels, use_time_emb=False, use_batchnorm=True, use_maxpool=False):
        super(ConvBlock, self).__init__()
        self.use_time_emb = use_time_emb
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        else:
            self.batchnorm = None
        if use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2)
        else:
            self.maxpool = None

        if self.use_time_emb:
            self.time_emb_proj = nn.Linear(time_emb_channels, out_channels)

    def forward(self, x, t_emb=None):
        x = self.conv(x)
        x = self.relu(x)

        if self.use_time_emb and t_emb is not None:
            t_emb_proj = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + t_emb_proj

        if self.batchnorm is not None:
            x = self.batchnorm(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        return x



class Encoder(nn.Module):
    def __init__(self, input_c=1, base_channel=16, time_emb_channels=128, num_classes=2):
        super(Encoder, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_channels)

        # 定义卷积层，确保正确传递 time_emb_channels
        # in_channels, out_channels, kernel_size, stride, padding, time_emb_channels, use_time_emb=False, use_batchnorm=True, use_maxpool=False
        self.conv1a = ConvBlock(input_c, base_channel, 5, 1, 2, time_emb_channels, use_time_emb=True, use_batchnorm=False, use_maxpool=False)
        self.conv1b = ConvBlock(base_channel, base_channel * 2, 5, 1, 2, time_emb_channels, use_time_emb=True, use_maxpool=True)
        self.conv2a = ConvBlock(base_channel * 2, base_channel * 4, 3, 1, 1, time_emb_channels, use_time_emb=True)
        self.conv2b = ConvBlock(base_channel * 4, base_channel * 4, 3, 1, 1, time_emb_channels, use_time_emb=True, use_maxpool=True)
        self.conv3a = ConvBlock(base_channel * 4, base_channel * 8, 3, 1, 1, time_emb_channels, use_time_emb=True)
        self.conv3b = ConvBlock(base_channel * 8, base_channel * 8, 3, 1, 1, time_emb_channels, use_time_emb=True, use_maxpool=True)

        # 计算全连接层的输入维度
        conv_output_dim = base_channel * 8 * 6 * 6  # 扁平化后的卷积输出维度
        fc_input_dim = conv_output_dim + time_emb_channels  # 加上嵌入的维度


        # 定义全连接层
        self.fc1 = nn.Sequential(nn.Linear(fc_input_dim, 32), nn.ReLU())
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, num_classes), nn.Sigmoid())

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        # 通过卷积层传递数据
        x = self.conv1a(x, t_emb)
        x = self.conv1b(x, t_emb)
        x = self.conv2a(x, t_emb)
        x = self.conv2b(x, t_emb)
        x = self.conv3a(x, t_emb)
        x = self.conv3b(x, t_emb)

        # 扁平化并与时间嵌入拼接
        x = x.view(x.size(0), -1)

        x = torch.cat([x, t_emb], dim=1)


        # 通过全连接层传递数据
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    def feature(self, x, t):
        t_emb = self.time_emb(t)

        # 通过卷积层传递数据
        x = self.conv1a(x, t_emb)
        x = self.conv1b(x, t_emb)
        x = self.conv2a(x, t_emb)
        x = self.conv2b(x, t_emb)
        x = self.conv3a(x, t_emb)
        x = self.conv3b(x, t_emb)

        # 扁平化并与时间嵌入拼接
        x = x.view(x.size(0), -1)
        x = torch.cat([x, t_emb], dim=1)
        # 通过全连接层传递数据
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


    





    
