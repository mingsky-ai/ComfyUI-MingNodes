import cv2
import numpy as np
import torch


def adjust_brightness_contrast_saturation(image, brightness, contrast, saturation):
    adjusted = cv2.convertScaleAbs(image, alpha = contrast, beta = brightness*60)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
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
                "brightness": ("FLOAT", {"default": 0, "min": -1.0, "max": 3.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0, "max": 3.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0, "max": 3.0, "step": 0.1}),
            }
        }

    CATEGORY = "MingNode/Image Process"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
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