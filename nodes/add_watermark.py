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
    # 调整水印大小
    width, height = original.size
    watermark = watermark.resize((int(watermark.width * scale), int(watermark.height * scale)), Image.LANCZOS)
    # 调整水印透明度
    alpha = watermark.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    watermark.putalpha(alpha)
    # 创建一个透明图层
    transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    # 将水印粘贴到透明图层上
    transparent.paste(watermark, (x, y), watermark)
    # 将透明图层与原始图片合并
    watermarked = Image.alpha_composite(original, transparent)
    return watermarked


def add_text_watermark(original, text, x, y, scale, opacity, color, fonts):
    # 创建一个透明图层
    txt = Image.new('RGBA', original.size, (255, 255, 255, 0))
    # 获取字体
    font_path = os.path.join(folder_paths.get_output_directory(), 'ComfyUI-MingNodes', 'fonts')
    font_path = font_path.replace("output", "custom_nodes")
    font_path = os.path.join(font_path, fonts)
    font_size = int(40 * scale)
    font = ImageFont.truetype(font_path, font_size)
    # 创建绘图对象
    d = ImageDraw.Draw(txt)
    # 添加文字水印
    d.text((x, y), text, font=font, fill=color + (int(255 * opacity),))
    # 将透明图层与原始图片合并
    watermarked = Image.alpha_composite(original, txt)
    return watermarked


def hex_to_rgb(hex_color):
    # 去掉可能包含的#号
    hex_color = hex_color.lstrip('#')
    # 将十六进制颜色转换为RGB格式
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
                "image_watermark_path": ("STRING",),
                "text": ("STRING",),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "fonts": ((sorted(files),)),
            },
        }

    CATEGORY = "MingNode/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_watermark"

    def add_watermark(self, image, image_watermark, position_X, position_Y, opacity, scale, image_watermark_path,
                      text, text_color, fonts):

        if image_watermark:
            result = []
            for img in image:
                img_cv1 = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
            img_cv2 = Image.open(image_watermark_path).convert("RGBA")
            adjusted_image = add_image_watermark(img_cv1, img_cv2, position_X, position_Y, opacity, scale)
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
