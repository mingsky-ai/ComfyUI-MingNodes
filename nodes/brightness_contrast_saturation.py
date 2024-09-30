from PIL import Image, ImageEnhance
import numpy as np
import torch


def adjust_image(img, brightness=0, contrast=0, saturation=0):

    if brightness != 0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1 + brightness / 10)
    
    if contrast != 0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1 + contrast / 10)
    
    if saturation != 0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1 + saturation / 10)
    
    return img


class AdjustBrightnessContrastSaturationNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "brightness_contrast_saturation"

    def brightness_contrast_saturation(self, image, brightness, contrast, saturation):
        for img in image:
            rgb_image = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
            adjusted_image = adjust_image(rgb_image, brightness, contrast, saturation)
            rst = torch.from_numpy(np.array(adjusted_image).astype(np.float32) / 255.0).unsqueeze(0)

        return (rst,)
