from PIL import Image
import numpy as np
import torch


def calculate_thresholds(luminance, shadow_percentile=45, highlight_percentile=55):
    hist, _ = np.histogram(luminance, bins=256, range=(0, 255))
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    shadow_threshold = np.argmin(np.abs(cdf_normalized - (shadow_percentile / 100 * cdf_normalized.max())))
    highlight_threshold = np.argmin(np.abs(cdf_normalized - (highlight_percentile / 100 * cdf_normalized.max())))
    return shadow_threshold, highlight_threshold


def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))


def adjust_color_balance(image, cyan_red, magenta_green, yellow_blue, adjustment_type='midtones'):
    img_array = np.array(image).astype(float)
    original_luminance = 0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2]
    shadow_threshold, highlight_threshold = calculate_thresholds(original_luminance)

    if adjustment_type == 'shadows':
        mask = sigmoid((shadow_threshold - original_luminance) / 20)
    elif adjustment_type == 'highlights':
        mask = sigmoid((original_luminance - highlight_threshold) / 20)
    else:
        min_luminance = np.min(original_luminance)
        max_luminance = np.max(original_luminance)
        mid_luminance = (min_luminance + max_luminance) / 2
        mask = 1 - np.abs(original_luminance - mid_luminance) / (max_luminance - min_luminance)
        mask = np.power(mask, 2)

    mask = np.stack([mask] * 3, axis=-1)
    adjustments = np.array([cyan_red, magenta_green, yellow_blue])
    adjusted_array = img_array + adjustments * mask
    adjusted_array = np.clip(adjusted_array, 0, 255)
    adjusted_luminance = 0.299 * adjusted_array[..., 0] + 0.587 * adjusted_array[..., 1] + 0.114 * adjusted_array[..., 2]
    luminance_ratio = original_luminance / (adjusted_luminance + 1e-8)

    for i in range(3):
        adjusted_array[..., i] *= luminance_ratio

    adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)
    return adjusted_array


class ColorBalanceNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "adjust_type": (
                    [
                        'midtones',
                        'highlights',
                        'shadows',
                    ],
                    {"default": "midtones"},
                ),
                "cyan_red": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "magenta_green": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "yellow_blue": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_balance"

    def color_balance(self, image, adjust_type, cyan_red, magenta_green, yellow_blue):
        for img in image:
            rgb_image = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        balanced_image = adjust_color_balance(rgb_image, cyan_red*1.5, magenta_green*1.5, yellow_blue*1.5, adjust_type)
        rst = torch.from_numpy(np.array(balanced_image).astype(np.float32) / 255.0).unsqueeze(0)
        return (rst,)
