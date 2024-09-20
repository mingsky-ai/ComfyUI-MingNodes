from .gray_channel  import ConvertGrayChannelNode
from .brightness_contrast_saturation  import AdjustBrightnessContrastSaturationNode
from .baidu_translate import BaiduTranslateNode


NODE_CLASS_MAPPINGS = {
    "ConvertGrayChannelNode": ConvertGrayChannelNode,
    "AdjustBrightnessContrastSaturationNode": AdjustBrightnessContrastSaturationNode,
    "BaiduTranslateNode": BaiduTranslateNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertGrayChannelNode": "Image To Grayscale And Channels",
    "AdjustBrightnessContrastSaturationNode": "Adjust Brightness Contrast Saturation",
    "BaiduTranslateNode": "Baidu Translate"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
