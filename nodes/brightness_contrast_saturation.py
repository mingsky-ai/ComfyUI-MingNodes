import cv2
import numpy as np
import torch


def adjust_contrast(image, contrast):
    cnst = 1
    if contrast > 1:
        cnst = contrast * 0.7
    elif contrast < 1:
        cnst = contrast
    beta = 0
    if contrast > 1:
        beta = np.mean(image) * 0.2 * (1 - contrast)
    elif contrast < 1:
        beta = np.mean(image) * 0.5 * (1 - contrast)
    adjusted = cv2.convertScaleAbs(image, alpha=cnst, beta=beta)
    if contrast > 1:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.addWeighted(s, contrast * 0.4, 0, 0, 0)
        vibrance = cv2.merge((h, s, v))
        return cv2.cvtColor(vibrance, cv2.COLOR_HSV2BGR)
    else:
        return adjusted


def adjust_brightness_contrast_saturation(image, brightness, contrast, saturation):
    blank = np.zeros_like(image)
    if brightness > 1:
        br = (brightness - 1) * 100
    elif brightness < 1:
        br = (brightness - 1) * 200
    else:
        br = 0
    adjusted = cv2.addWeighted(image, 1, blank, 0.5, br)
    adjusted2 = adjust_contrast(adjusted, contrast)
    hsv = cv2.cvtColor(adjusted2, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.addWeighted(s, saturation, 0, 0, 0)
    vibrance = cv2.merge((h, s, v))
    return cv2.cvtColor(vibrance, cv2.COLOR_HSV2BGR)


class AdjustBrightnessContrastSaturationNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.01}),
            }
        }

    CATEGORY = "MingNode/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "brightness_contrast_saturation"

    def brightness_contrast_saturation(self, image, brightness, contrast, saturation):
        result = []
        for img in image:
            img_cv2 = img.cpu().numpy()
            img_cv2 = (img_cv2 * 255).astype(np.uint8)
            adjusted_image = adjust_brightness_contrast_saturation(img_cv2, brightness, contrast, saturation)
            rst = torch.from_numpy(adjusted_image.astype(np.float32) / 255.0).to(image.device)
            result.append(rst)
        final_tensor = torch.stack(result)

        return (final_tensor,)
