import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import folder_paths
import os


def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def add_image_watermark(original, watermark, x, y, opacity, scale):
    width, height = original.size
    watermark = watermark.resize((int(watermark.width * scale), int(watermark.height * scale)), Image.LANCZOS)
    alpha = watermark.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    watermark.putalpha(alpha)
    transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    transparent.paste(watermark, (x, y), watermark)
    watermarked = Image.alpha_composite(original, transparent)
    return watermarked


def add_text_watermark(original, text, x, y, scale, opacity, color, fonts):
    txt = Image.new('RGBA', original.size, (255, 255, 255, 0))
    font_path = os.path.join(folder_paths.get_output_directory(), 'ComfyUI-MingNodes', 'fonts')
    font_path = font_path.replace("output", "custom_nodes")
    font_path = os.path.join(font_path, fonts)
    font_size = int(40 * scale)
    font = ImageFont.truetype(font_path, font_size)
    d = ImageDraw.Draw(txt)
    d.text((x, y), text, font=font, fill=color + (int(255 * opacity),))
    watermarked = Image.alpha_composite(original, txt)
    return watermarked


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


class AddWaterMarkNode:
    @classmethod
    def INPUT_TYPES(s):
        font_path = os.path.join(folder_paths.get_output_directory(), 'ComfyUI-MingNodes', 'fonts')
        font_path = font_path.replace("output", "custom_nodes")
        files = [f for f in os.listdir(font_path) if os.path.isfile(os.path.join(font_path, f))]

        return {
            "required": {
                "image": ("IMAGE",),
                "image_watermark": ("BOOLEAN", {"default": True}),
                "position_X": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "position_Y": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "watermark": ("IMAGE",),
                "watermark_mask": ("MASK",),
                "text": ("STRING", {"default": "enter text"}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "fonts": ((sorted(files),)),
            },
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_watermark"

    def add_watermark(self, image, image_watermark, position_X, position_Y, opacity, scale,
                      text, text_color, fonts, watermark=None, watermark_mask=None,):

        if image_watermark:
            result = []
            for img1 in image:
                img_cv1 = Image.fromarray((img1.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
            for img2 in watermark:
                rgb_image = Image.fromarray((img2.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
            for img3 in watermark_mask:
                mask_image = Image.fromarray((img3.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("L")
            if rgb_image.size != mask_image.size:
                raise ValueError("RGB图像和mask图像的大小必须相同")
            rgba_image = Image.new('RGBA', rgb_image.size)
            rgb_data = rgb_image.getdata()
            mask_data = mask_image.getdata()
            rgba_data = []
            for rgb, alpha in zip(rgb_data, mask_data):
                inverted_alpha = 255 - alpha
                rgba_data.append(rgb + (inverted_alpha,))
            rgba_image.putdata(rgba_data)
            adjusted_image = add_image_watermark(img_cv1, rgba_image, position_X, position_Y, opacity, scale)
            rst = torch.from_numpy(np.array(adjusted_image).astype(np.float32) / 255.0).to(image.device)
            result.append(rst)
            final_tensor = torch.stack(result)
            return (final_tensor,)
        else:
            result2 = []
            for img in image:
                img_cv2 = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
            adjusted_image2 = add_text_watermark(img_cv2, str(text), position_X, position_Y, scale,
                                                 opacity, hex_to_rgb(text_color), fonts)
            rst2 = torch.from_numpy(np.array(adjusted_image2).astype(np.float32) / 255.0).to(image.device)
            result2.append(rst2)
            final_tensor2 = torch.stack(result2)
            return (final_tensor2,)
