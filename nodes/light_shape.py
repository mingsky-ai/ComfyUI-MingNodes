from PIL import Image, ImageDraw, ImageFilter
import math
import numpy as np
import torch


def draw_shape(shape, size=(200, 200), offset=(0, 0), scale=1.0, rotation=0, bg_color=(255, 255, 255),
               shape_color=(0, 0, 0), opacity=1.0, blur_radius=0, base_image=None):
    width, height = size
    offset_x, offset_y = offset
    center_x, center_y = width // 2 + offset_x, height // 2 + offset_y
    max_dim = min(width, height) * scale

    diagonal = int(math.sqrt(width ** 2 + height ** 2))
    img_tmp = Image.new('RGBA', (diagonal, diagonal), (0, 0, 0, 0))
    draw_tmp = ImageDraw.Draw(img_tmp)

    tmp_center = diagonal // 2

    alpha = int(opacity * 255)
    shape_color = shape_color + (alpha,)

    if shape == 'circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'semicircle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=180, fill=shape_color)

    elif shape == 'quarter_circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=90, fill=shape_color)

    elif shape == 'ellipse':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 4, tmp_center + max_dim / 2, tmp_center + max_dim / 4)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'square':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.rectangle(bbox, fill=shape_color)

    elif shape == 'triangle':
        points = [
            (tmp_center, tmp_center - max_dim / 2),
            (tmp_center - max_dim / 2, tmp_center + max_dim / 2),
            (tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        ]
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'cross':
        vertical = [(tmp_center - max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center + max_dim / 2),
                    (tmp_center - max_dim / 6, tmp_center + max_dim / 2)]
        horizontal = [(tmp_center - max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center + max_dim / 6),
                      (tmp_center - max_dim / 2, tmp_center + max_dim / 6)]
        draw_tmp.polygon(vertical, fill=shape_color)
        draw_tmp.polygon(horizontal, fill=shape_color)

    elif shape == 'star':
        points = []
        for i in range(10):
            angle = i * 36 * math.pi / 180
            radius = max_dim / 2 if i % 2 == 0 else max_dim / 4
            points.append((tmp_center + radius * math.sin(angle), tmp_center - radius * math.cos(angle)))
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'radial':
        num_rays = 12
        for i in range(num_rays):
            angle = i * (360 / num_rays) * math.pi / 180
            x1 = tmp_center + max_dim / 4 * math.cos(angle)
            y1 = tmp_center + max_dim / 4 * math.sin(angle)
            x2 = tmp_center + max_dim / 2 * math.cos(angle)
            y2 = tmp_center + max_dim / 2 * math.sin(angle)
            draw_tmp.line([(x1, y1), (x2, y2)], fill=shape_color, width=int(max_dim / 20))

    img_tmp = img_tmp.rotate(rotation, resample=Image.BICUBIC, expand=True)
    if base_image is None:
        img = Image.new('RGBA', size, bg_color + (255,))
    else:
        img = base_image.copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

    paste_x = center_x - img_tmp.width // 2
    paste_y = center_y - img_tmp.height // 2

    img.alpha_composite(img_tmp, (paste_x, paste_y))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


class LightShapeNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wide": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "shape": (
                    [
                        'circle',
                        'square',
                        'semicircle',
                        'quarter_circle',
                        'ellipse',
                        'triangle',
                        'cross',
                        'star',
                        'radial',
                    ],
                    {"default": "circle"},
                ),
                "X_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "Y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "background_color": ("STRING", {"default": "#000000"}),
                "shape_color": ("STRING", {"default": "#FFFFFF"}),
            },
            "optional": {
                "base_image": ("IMAGE", {"default": None}),
            },
        }

    CATEGORY = "MingNodes/Image Process"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "drew_light_shape"

    def drew_light_shape(self, wide, height, shape, X_offset, Y_offset, scale, rotation, opacity, blur_radius, background_color,
                         shape_color, base_image=None):

        if base_image is None:
            img = draw_shape(shape, size=(wide, height), offset=(X_offset, Y_offset), scale=scale,
                             rotation=rotation,
                             bg_color=hex_to_rgb(background_color), shape_color=hex_to_rgb(shape_color),
                             opacity=opacity, blur_radius=blur_radius)
        else:
            img_cv = Image.fromarray((base_image.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
            img = draw_shape(shape, size=(wide, height), offset=(X_offset, Y_offset), scale=scale,
                             rotation=rotation,
                             bg_color=hex_to_rgb(background_color), shape_color=hex_to_rgb(shape_color),
                             opacity=opacity, blur_radius=blur_radius, base_image=img_cv)

        rst = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return (rst,)
