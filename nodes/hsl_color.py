import cv2
import numpy as np
import torch


def create_color_mask(hsv_image, color_range):
    mask = np.zeros(hsv_image.shape[:2], dtype=np.float32)
    h, s, v = cv2.split(hsv_image)

    for lower, upper in color_range:
        lower_h, lower_s, lower_v = lower
        upper_h, upper_s, upper_v = upper
        if lower_h > upper_h:
            h_mask = ((h >= lower_h) | (h <= upper_h)).astype(np.float32)
        else:
            h_mask = ((h >= lower_h) & (h <= upper_h)).astype(np.float32)
        s_mask = ((s >= lower_s) & (s <= upper_s)).astype(np.float32)
        v_mask = ((v >= lower_v) & (v <= upper_v)).astype(np.float32)
        combined_mask = (h_mask * 0.8 + s_mask * 0.15 + v_mask * 0.05)
        mask = np.maximum(mask, combined_mask)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5)

    return mask


def adjust_hsl(image, color_adjustments):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    color_ranges = {
        'red': [([0, 70, 70], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])],
        'orange': [([11, 70, 70], [20, 255, 255])],
        'yellow': [([21, 70, 70], [30, 255, 255])],
        'green': [([31, 70, 70], [75, 255, 255])],
        'lightGreen': [([76, 70, 70], [90, 255, 255])],
        'blue': [([96, 50, 50], [130, 255, 255])],
        'purple': [([131, 70, 70], [155, 255, 255])],
        'magenta': [([156, 70, 70], [169, 255, 255])]
    }

    all_masks = {}
    for color, ranges in color_ranges.items():
        all_masks[color] = create_color_mask(hsv_image, ranges)

    adjustment_mask = np.zeros_like(hsv_image)

    for color, (h_shift, s_shift, l_shift) in color_adjustments.items():
        color_mask = all_masks[color]
        if np.max(color_mask) < 1:
            continue
        if h_shift != 0:
            hue_multiplier = 2 if color == 'blue' else 1
            adjustment_mask[:, :, 0] += h_shift * color_mask * hue_multiplier
        if s_shift != 0:
            adjustment_mask[:, :, 1] += s_shift * color_mask * 3
        if l_shift != 0:
            adjustment_mask[:, :, 2] += l_shift * color_mask * 2

    for i in range(3):
        adjustment_mask[:, :, i] = cv2.GaussianBlur(adjustment_mask[:, :, i], (0, 0), sigmaX=3, sigmaY=3)

    hsv_image[:, :, 0] = np.mod(hsv_image[:, :, 0] + adjustment_mask[:, :, 0], 180)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + adjustment_mask[:, :, 1], 0, 255)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + adjustment_mask[:, :, 2], 0, 255)

    adjusted_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted_image


def create_color_adjustments(red, orange, yellow, green, lightGreen, blue, purple, magenta):
    return {
        'red': red,
        'orange': orange,
        'yellow': yellow,
        'green': green,
        'lightGreen': lightGreen,
        'blue': blue,
        'purple': purple,
        'magenta': magenta
    }


def tensor2cv2(image: torch.Tensor) -> np.array:
    if image.dim() == 4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)


class HSLColorNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "red_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "red_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "orange_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "orange_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "orange_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "yellow_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "yellow_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "yellow_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "green_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "green_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "green_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "lightGreen_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "lightGreen_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "lightGreen_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "blue_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "blue_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "blue_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "purple_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "purple_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "purple_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "magenta_hue": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "magenta_saturation": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "magenta_brightness": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
            }
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "hsl_color"

    def hsl_color(self, image, red_hue, red_saturation, red_brightness, orange_hue, orange_saturation, orange_brightness
                  , yellow_hue, yellow_saturation, yellow_brightness, green_hue, green_saturation, green_brightness
                  , lightGreen_hue, lightGreen_saturation, lightGreen_brightness, blue_hue, blue_saturation,
                  blue_brightness, purple_hue, purple_saturation, purple_brightness, magenta_hue, magenta_saturation,
                  magenta_brightness):
        for img in image:
            img_cv1 = tensor2cv2(img)
        color_adjustments = create_color_adjustments(
            (red_hue, red_saturation, red_brightness),
            (orange_hue, orange_saturation, orange_brightness),
            (yellow_hue, yellow_saturation, yellow_brightness),
            (green_hue, green_saturation, green_brightness),
            (lightGreen_hue, lightGreen_saturation, lightGreen_brightness),
            (blue_hue, blue_saturation, blue_brightness),
            (purple_hue, purple_saturation, purple_brightness),
            (magenta_hue, magenta_saturation, magenta_brightness),
        )
        adjusted_image = adjust_hsl(img_cv1, color_adjustments)
        result_img = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
        rst = torch.from_numpy(result_img.astype(np.float32) / 255.0).unsqueeze(0)
        return (rst,)
