import numpy as np
import cv2
import torch


def image_stats(image):
    means = []
    stds = []
    for channel in range(1, 3):
        means.append(np.mean(image[:, :, channel]))
        stds.append(np.std(image[:, :, channel]))
    return means, stds


def is_skin(l, a, b):
    return (l > 20) and (l < 220) and \
           (a > 130) and (a < 170) and \
           (b > 130) and (b < 180)


def tensor2cv2(image: torch.Tensor) -> np.array:
    if image.dim() == 4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)


def color_transfer(source, target, strength=0.8, skin_protection=0.7):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    target_l, target_a, target_b = cv2.split(target_lab)
    src_means, src_stds = image_stats(source_lab)
    tar_means, tar_stds = image_stats(target_lab)

    skin_mask = np.apply_along_axis(lambda x: is_skin(*x), 2, target_lab).astype(np.float32)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    for i, channel in enumerate([target_a, target_b]):
        channel = channel.astype(np.float32)
        adjusted_channel = (channel - tar_means[i]) * (src_stds[i] / tar_stds[i]) + src_means[i]
        skin_adjusted = channel * skin_protection + adjusted_channel * (1 - skin_protection)
        non_skin_adjusted = channel * (1 - strength) + adjusted_channel * strength
        channel = skin_mask * skin_adjusted + (1 - skin_mask) * non_skin_adjusted
        if i == 0:
            target_a = np.clip(channel, 0, 255).astype(np.uint8)
        else:
            target_b = np.clip(channel, 0, 255).astype(np.uint8)

    target_l = cv2.addWeighted(target_l, 0.95, cv2.GaussianBlur(target_l, (5, 5), 0), 0.05, 0)

    result_lab = cv2.merge([target_l, target_a, target_b])
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    return result_bgr


class ImitationHueNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imitation_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1, "min": 0.1, "max": 1.0, "step": 0.1}),
                "skin_protection": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            }
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "imitation_hue"

    def imitation_hue(self, imitation_image, target_image, strength, skin_protection):
        for img in imitation_image:
            img_cv1 = tensor2cv2(img)

        for img in target_image:
            img_cv2 = tensor2cv2(img)

        result_img = color_transfer(img_cv1, img_cv2, strength, skin_protection)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        rst = torch.from_numpy(result_img.astype(np.float32) / 255.0).unsqueeze(0)

        return (rst,)
