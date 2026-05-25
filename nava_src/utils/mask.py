import torch

import os
import numpy as np
import torch

def save_bias_to_txt(bias: torch.Tensor, tag: str = "bias_debug"):
    """
    把 attention bias 存成 txt，方便排查。
    假设 bias 形状为 [B, L, L]，比如 [2, 441, 441]。
    如果是 [B, 1, L, L] 或 [B, H, L, L] 也可以先 squeeze / 选一个 head 再传进来。

    参数：
      bias: torch.Tensor, [B, L, L]
      tag:  文件名后缀，用来区分不同 step/batch，比如 'step_100'
    """
    # 你可以根据需要改存放路径
    save_dir = "/root/baidu/personal-code/sy_transfusion"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bias_{tag}.txt")

    with open(save_path, "w") as f:
        f.write("=== attention bias debug ===\n")

        if bias is None:
            f.write("bias is None\n")
            return

        # 移到 CPU，转 numpy
        bias_np = bias.detach().cpu().numpy()
        f.write(f"shape: {bias_np.shape}\n")

        # 不要省略打印
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        B = bias_np.shape[0]
        for b in range(B):
            f.write(f"\n[batch {b}]\n")
            # 这一行是一个 [L, L] 矩阵
            np.savetxt(f, bias_np[b], fmt="%.2f")  # 你可以改精度
    print(f"[DEBUG] bias saved to {save_path}")


def make_transfusion_attention_mask(
    batch_spans, 
    L: int, 
    device=None, 
    dtype=None,
    seq_valid_mask: torch.Tensor | None = None,
    ):
    
    """
    构造加性 bias：
      - 默认：因果 (只看自己和过去)
      - 图像 span 内：双向解锁
      - seq_valid_mask: [B, L]，1=有效 token，0=padding
        -> padding 作为 key 一律 -inf
    返回 [B, 1, L, L] 形状（方便广播到多头）。
    """
    B = len(batch_spans)
    bias = torch.zeros((B, L, L), device=device, dtype=dtype)
    # 1) 因果：只能看过去和当前
    i = torch.arange(L, device=device)
    future = i[None, :, None] < i[None, None, :]
    bias.masked_fill_(future, float("-inf"))

    # 2) 图像 span 内双向解锁：把这个子块重新改成 0
    #    注意：这里仅解除因果约束，不处理 padding；
    #    padding 的行/列后面会统一再 mask 掉。
    for b, spans in enumerate(batch_spans):
        for (s, e) in spans:
            if s is None or e is None:
                continue
            s_clamp = max(0, int(s))
            e_clamp = min(L - 1, int(e))
            if e_clamp < s_clamp:
                continue
            # span 内 [s:e] × [s:e] 全 0（双向）
            bias[b, s_clamp:e_clamp+1, s_clamp:e_clamp+1] = 0.0

    # 3) padding 屏蔽：同时作为 key 和 query
    if seq_valid_mask is not None:
        valid = seq_valid_mask.to(device=device).bool()  # [B,L]
        invalid = ~valid                                 # [B,L]

        # (a) 作为 key：别人看它 → 屏蔽列
        key_invalid = invalid.unsqueeze(1).expand(-1, L, -1)   # [B,L,L]
        bias = bias.masked_fill(key_invalid, float("-inf"))

        # (b) 作为 query：它看别人 → 屏蔽行
        # query_invalid = invalid.unsqueeze(2).expand(-1, -1, L) # [B,L,L]
        # bias = bias.masked_fill(query_invalid, float("-inf"))

        # # (c) 防止整行全是 -inf：对 padding token 自己留一个 0
        # for b in range(B):
        #     pad_idx = torch.nonzero(invalid[b], as_tuple=False).flatten()
        #     bias[b, pad_idx, pad_idx] = 0.0

    # print(batch_spans)
    # print("DEBUG 88888:")
    # print(bias.shape)
    # save_bias_to_txt(bias)

    return bias.unsqueeze(1)    # [B,1,L,L]
