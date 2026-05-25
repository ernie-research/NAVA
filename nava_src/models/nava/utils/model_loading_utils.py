import torch 
import os
import json
from safetensors.torch import load_file

from nava_src.models.nava.modules.fusion import FusionModel
from nava_src.models.nava.modules.t5 import T5EncoderModel
from nava_src.models.nava.modules.vae2_2 import Wan2_2_VAE
    
def init_wan_vae_2_2(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['device'] = rank
    vae_pth = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae_config['vae_pth'] = vae_pth
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model

def init_fusion_score_model_ovi(rank: int = 0, meta_init=False):
    video_config = "ovi/configs/model/dit/video.json"
    audio_config = "ovi/configs/model/dit/audio.json"
    assert os.path.exists(video_config), f"{video_config} does not exist"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config

def init_text_model(ckpt_dir, rank, cpu_offload=False):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=cpu_offload,
        shard_fn=None)


    return text_encoder



def load_fusion_checkpoint(model, checkpoint_path, from_meta=False, device="cpu"):
    assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist"

    # =============== 2. 从 checkpoint 加载 ===============
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"{checkpoint_path=} does not exist")

    if checkpoint_path and os.path.exists(checkpoint_path):
        # copy a params from fusion model to single model key
        df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
        for key in model.state_dict().keys():
            if "fusion_blocks" in key:
                if "vid_block" in key:
                    layer_idx = key.split(".")[2]
                    model_struc = key.split("vid_block.")[-1]
                    supp_key = f"backbone.video_model.blocks.{layer_idx}.{model_struc}"
                    if supp_key in model.state_dict():
                        df[supp_key] = df[key]
                elif "audio_block" in key:
                    layer_idx = key.split(".")[2]
                    model_struc = key.split("audio_block.")[-1]
                    supp_key = f"backbone.audio_model.blocks.{layer_idx}.{model_struc}"
                    if supp_key in model.state_dict():
                        df[supp_key] = df[key]

        missing, unexpected = model.load_state_dict(df, strict=True, assign=from_meta)
        print(missing, unexpected)

        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else: 
        raise RuntimeError("{checkpoint=} does not exists'")