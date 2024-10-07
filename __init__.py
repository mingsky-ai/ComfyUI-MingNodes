from .nodes.gray_channel import ConvertGrayChannelNode
from .nodes.brightness_contrast_saturation import AdjustBrightnessContrastSaturationNode
from .nodes.baidu_translate import BaiduTranslateNode
from .nodes.add_watermark import AddWaterMarkNode
from .nodes.imitation_hue import ImitationHueNode
from .nodes.light_shape import LightShapeNode
from .nodes.highlight_shadow_brightness import HighlightShadowBrightnessNode
from .nodes.hsl_color import HSLColorNode

NODE_CLASS_MAPPINGS = {
    "ConvertGrayChannelNode": ConvertGrayChannelNode,
    "AdjustBrightnessContrastSaturationNode": AdjustBrightnessContrastSaturationNode,
    "BaiduTranslateNode": BaiduTranslateNode,
    "AddWaterMarkNode": AddWaterMarkNode,
    "LightShapeNode": LightShapeNode,
    "ImitationHueNode": ImitationHueNode,
    "HighlightShadowBrightnessNode": HighlightShadowBrightnessNode,
    "HSLColorNode": HSLColorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertGrayChannelNode": "Grayscale Channels",
    "AdjustBrightnessContrastSaturationNode": "Brightness Contrast Saturation",
    "BaiduTranslateNode": "Baidu Translate",
    "AddWaterMarkNode": "Add Watermark",
    "LightShapeNode": "IC-Light Light Shape",
    "ImitationHueNode": "Imitation Hue",
    "HighlightShadowBrightnessNode": "Highlight Shadow Brightness",
    "HSLColorNode": "HSL Color",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
