import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

_MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
_instance = None

def get_enhancer(gpu_id):
    """Lazy-init a shared RealESRGANer instance (downloads weights on first call)."""
    global _instance
    if _instance is not None:
        return _instance

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4,
    )

    if gpu_id == 'mps':
        device = torch.device('mps')
    elif gpu_id != '-1':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    _instance = RealESRGANer(
        scale=4,
        model_path=_MODEL_URL,
        model=model,
        tile=0,
        pre_pad=10,
        half=False,
        device=device,
    )
    return _instance


def enhance_patch(img_bgr, gpu_id, outscale=1):
    """Enhance a small BGR image patch using Real-ESRGAN.

    outscale=1 means output same size as input (upscale 4x then downscale).
    outscale=2 means output 2x the input size, etc.
    """
    upsampler = get_enhancer(gpu_id)
    try:
        output, _ = upsampler.enhance(img_bgr, outscale=outscale)
    except RuntimeError:
        output = img_bgr
    return output
