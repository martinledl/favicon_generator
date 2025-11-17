import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        if use_attention:
            self.attn = SelfAttentionBlock(out_channels)

    def forward(self, x, t_emb):
        h = self.block1(x)
        t = self.time_mlp(t_emb).view(x.shape[0], -1, 1, 1)
        h = h + t
        h = self.block2(h)

        if self.use_attention:
            h = self.attn(h)

        return h + self.res_conv(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.view(B, C, H * W)  # (B, C, N)
        qkv = self.qkv(x)  # (B, 3C, N)
        q, k, v = qkv.chunk(3, dim=1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)  # (B, N, N)
        out = v @ attn.transpose(1, 2)  # (B, C, N)
        out = self.proj(out)
        return (x + out).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]  # (B, half)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # (B, dim)


class UNet(nn.Module):
    def __init__(self, num_adjectives, num_industries, in_channels=3, base_channels=32, time_emb_dim=128,
                 cond_emb_dim=128):
        super().__init__()

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Embeddings
        self.adjective_emb = nn.Embedding(num_adjectives, cond_emb_dim // 2, padding_idx=-1)
        self.industry_emb = nn.Embedding(num_industries, cond_emb_dim // 2, padding_idx=-1)
        self.cond_mapper = nn.Sequential(
            nn.Linear(cond_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.res1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = Downsample(base_channels)

        self.res2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = Downsample(base_channels * 2)

        # Bottleneck
        self.mid = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        # Decoder
        self.up1 = Upsample(base_channels * 2)
        self.res3 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_emb_dim)

        self.up2 = Upsample(base_channels)
        self.res4 = ResidualBlock(base_channels + base_channels, base_channels, time_emb_dim)

        self.conv_out = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, adjective_id, industry_id):
        # time embedding
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # condition embedding
        adj_emb = self.adjective_emb(adjective_id)
        ind_emb = self.industry_emb(industry_id)
        y_emb = torch.cat([adj_emb, ind_emb], dim=1)
        y_emb = self.cond_mapper(y_emb)

        emb = t_emb + y_emb

        x1 = self.conv_in(x)
        x1r = self.res1(x1, emb)

        x2 = self.down1(x1r)
        x2r = self.res2(x2, emb)

        x3 = self.down2(x2r)
        x3 = self.mid(x3, emb)

        x4 = self.up1(x3)
        x4 = torch.cat([x4, x2r], dim=1)
        x4 = self.res3(x4, emb)

        x5 = self.up2(x4)
        x5 = torch.cat([x5, x1r], dim=1)
        x5 = self.res4(x5, emb)

        return self.conv_out(x5)


def q_sample(x, t, epsilon, alphas_bar):
    """
    x: clean image (B, C, H, W)
    t: timestep tensor (B,)
    epsilon: noise (B, C, H, W)
    alphas_bar: precomputed ᾱ_t schedule (T,)
    """
    sqrt_alpha_bar = torch.sqrt(alphas_bar[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_bar[t]).view(-1, 1, 1, 1)
    return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * epsilon


def cosine_beta_schedule(T, s=0.008):
    steps = T+1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x/T + s)/(1+s)) * math.pi/2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)