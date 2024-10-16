from litelama import LiteLama
from litelama.model import download_file
import os
from PIL import Image
import numpy as np
import folder_paths
import torch


def remove(image, mask):
    Lama = LiteLama2()
    device = "cuda:0"
    result = None
    try:
        Lama.to(device)
        result = Lama.predict(image, mask)
    except:
        pass
    finally:
        Lama.to("cpu")

    return result


class LiteLama2(LiteLama):
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self, checkpoint_path=None, config_path=None):
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._model = None

        if self._checkpoint_path is None:
            MODEL_PATH = "ming/"
            checkpoint_path = os.path.join(folder_paths.models_dir, MODEL_PATH, "big-lama.safetensors")
            if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
                pass
            else:
                download_file("https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors",
                              checkpoint_path)

            self._checkpoint_path = checkpoint_path

        self.load(location="cuda:0")


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class RemoveWatermarkNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_watermark"

    def remove_watermark(self, image, mask):
        for img in image:
            o_img = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        for ms in mask:
            m_img = Image.fromarray((ms.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        rm_img = remove(o_img, m_img)
        rst = pil2tensor(rm_img)
        return (rst,)
