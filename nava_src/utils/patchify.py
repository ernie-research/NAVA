from einops import rearrange
import torch
import torch.nn as nn

class LinearPatchProjector(nn.Module):
    """
    z: [B, C, H, W]  ->  patches: [B, (H/k)*(W/k), C*k*k]  ->  embeds: [B, N, d_model]
    """
    def __init__(self, latent_ch: int, k: int, d_model: int):
        super().__init__()
        self.k = k
        self.patch_dim = latent_ch * k * k
        # 用 fp32 算 Linear，数值更稳
        self.proj = nn.Linear(self.patch_dim, d_model, bias=True)


    def forward(self, z):  # z: [B,C,H,W]
        B, C, H, W = z.shape
        k = self.k
        assert H % k == 0 and W % k == 0, f"H,W 必须是 k 的整数倍, got {(H,W)} vs k={k}"
        # 统一用明确的 block 方式切 patch（不会踩 stride 坑）
        patches = rearrange(z, 'b c (hs k1) (ws k2) -> b (hs ws) (c k1 k2)', k1=k, k2=k)
        embeds = self.proj(patches)  # fp32 线性
        return embeds
