from .nodes.gray_channel  import ConvertGrayChannelNode
from .nodes.brightness_contrast_saturation  import AdjustBrightnessContrastSaturationNode
from .nodes.baidu_translate import BaiduTranslateNode


NODE_CLASS_MAPPINGS = {
    "ConvertGrayChannelNode": ConvertGrayChannelNode,
    "AdjustBrightnessContrastSaturationNode": AdjustBrightnessContrastSaturationNode,
    "BaiduTranslateNode": BaiduTranslateNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertGrayChannelNode": "Image To Grayscale And Channels",
    "AdjustBrightnessContrastSaturationNode": "Adjust Brightness Contrast Saturation",
    "BaiduTranslateNode": "Baidu Translate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
