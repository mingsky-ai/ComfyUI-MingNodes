from .nodes.gray_channel import ConvertGrayChannelNode
from .nodes.brightness_contrast_saturation import AdjustBrightnessContrastSaturationNode
from .nodes.baidu_translate import BaiduTranslateNode
from .nodes.add_watermark import AddWaterMarkNode
from .nodes.upload_image_path import LoadImagePathNode
from .nodes.light_shape import LightShapeNode
NODE_CLASS_MAPPINGS = {
    "ConvertGrayChannelNode": ConvertGrayChannelNode,
    "AdjustBrightnessContrastSaturationNode": AdjustBrightnessContrastSaturationNode,
    "BaiduTranslateNode": BaiduTranslateNode,
    "AddWaterMarkNode": AddWaterMarkNode,
    "LoadImagePathNode": LoadImagePathNode,
    "LightShapeNode": LightShapeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertGrayChannelNode": "Grayscale Channels",
    "AdjustBrightnessContrastSaturationNode": "Brightness Contrast Saturation",
    "BaiduTranslateNode": "Baidu Translate",
    "AddWaterMarkNode": "Add Watermark",
    "LoadImagePathNode": "Load Image",
    "LightShapeNode": "Draw IC-Light Light Shape",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
