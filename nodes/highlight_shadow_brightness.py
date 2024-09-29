from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import torch


def rgb_to_hsv(rgb):
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    hsv[:, :, 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[:, :, 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[:, :, 0] = (hsv[:, :, 0] / 6.0) % 1.0
    return hsv


def hsv_to_rgb(hsv):
    rgb = np.zeros_like(hsv)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[:, :, 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[:, :, 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[:, :, 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb


def adjust_image(img, highlight=0, shadow=0, brightness=0):
    img_array = np.array(img, dtype=float) / 255.0
    hsv = rgb_to_hsv(img_array)
    v = hsv[:, :, 2]
    if shadow != 0:
        shadow_mask = np.clip(1 - (v / 0.8), 0, 1)
        adjustment = shadow / 100
        delta = v * adjustment * shadow_mask
        v_adjusted = v + delta
        v = np.clip(v_adjusted, 0, 1)
    if highlight != 0:
        adjustment = highlight / 100
        highlight_threshold = 0.6
        highlight_mask = np.clip((v - highlight_threshold) / (1 - highlight_threshold), 0, 1)

        if adjustment < 0:
            v_compressed = v ** (1 - adjustment)
            v = v * (1 - highlight_mask) + v_compressed * highlight_mask
            kernel = np.ones((5, 5)) / 25
            local_mean = convolve2d(v, kernel, mode='same', boundary='symm')
            v_detail = v - local_mean
            v += v_detail * (-adjustment * 0.5)
            v = (v - v.min()) / (v.max() - v.min())
            v = np.power(v, 1 + (-adjustment * 0.2))
        else:
            v_adjusted = v + (1 - v) * adjustment * highlight_mask
            v = v_adjusted

    if brightness != 0:
        v *= (1 + brightness / 100)

    hsv[:, :, 2] = np.clip(v, 0, 1)
    adjusted_array = hsv_to_rgb(hsv)
    adjusted_array = (adjusted_array * 255).astype(np.uint8)

    return adjusted_array


class HighlightShadowBrightnessNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "highlight": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "shadow": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "brightness": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "highlight_shadow_brightness"

    def highlight_shadow_brightness(self, image, highlight, shadow, brightness):
        for img in image:
            rgb_image = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        result_image = adjust_image(rgb_image, highlight*10, shadow*10, brightness*10)
        rst = torch.from_numpy(result_image.astype(np.float32) / 255.0).unsqueeze(0)
        return (rst,)
