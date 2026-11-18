import torch
import torch.nn as nn

class PatchTSTRegressor(nn.Module):
    def __init__(self, n_channels, seq_len, patch_len, patch_stride, d_model, n_heads,
                 n_layers, d_ff, dropout, n_targets=3, bias_init=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_patches = (seq_len - patch_len) // patch_stride + 1
        idx = torch.arange(self.n_patches)[:, None] * patch_stride + torch.arange(patch_len)[None, :]
        self.register_buffer("patch_idx", idx, persistent=False)
        self.proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(n_channels * d_model),
            nn.Linear(n_channels * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )
        if bias_init is not None:
            with torch.no_grad():
                self.head[-1].bias.copy_(torch.as_tensor(bias_init))

    def forward(self, x):
        B, C, _ = x.shape
        p = x[:, :, self.patch_idx]
        p = p.reshape(B * C, self.n_patches, -1)
        h = self.drop(self.proj(p) + self.pos)
        h = self.encoder(h).mean(dim=1)
        return self.head(h.reshape(B, C * h.shape[-1]))
