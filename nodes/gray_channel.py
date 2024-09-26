import cv2
import numpy as np
import torch


class ConvertGrayChannelNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("grayscale", "blue channel", "green channel", "red channel",)
    FUNCTION = "convert_gray_channel"

    def convert_gray_channel(self, image):
        gray_images = []
        bb_images = []
        gg_images = []
        rr_images = []
        for img in image:
            img_cv2 = img.cpu().numpy()
            img_cv2 = (img_cv2 * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
            gy = torch.from_numpy(gray.astype(np.float32) / 255.0).to(image.device)
            gray_images.append(gy)

            sp = cv2.split(img_cv2)
            b = sp[0][:, :]
            bb = torch.from_numpy(b.astype(np.float32) / 255.0).to(image.device)
            bb_images.append(bb)

            g = sp[1][:, :]
            gg = torch.from_numpy(g.astype(np.float32) / 255.0).to(image.device)
            gg_images.append(gg)

            r = sp[2][:, :]
            rr = torch.from_numpy(r.astype(np.float32) / 255.0).to(image.device)
            rr_images.append(rr)

        final_tensor1 = torch.stack(gray_images)
        final_tensor2 = torch.stack(bb_images)
        final_tensor3 = torch.stack(gg_images)
        final_tensor4 = torch.stack(rr_images)

        return (final_tensor1, final_tensor2, final_tensor3, final_tensor4, )


