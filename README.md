# ComfyUI-MingNodes

开发的一些有用的功能，具体工作流在workflow文件夹

Some useful  functions developed, the specific workflow is in the workflow folder

1: 模仿像素蛋糕追色的功能，参数强度是色彩迁移的强度，皮肤保护参数值皮肤色彩越大越接近原图色彩,支持区域色彩迁移，加入SAM语义分割制作蒙板即可。
可以打开是否调节自动亮度，对比度，和饱和度选项，各项值的参数就是自动调节的范围0.5就是50%范围内自动调节。新增影调开关，可以模仿原图的影调。 
还有需要注意的是，如果是相机的原片JPG或PNG等图片，需要到PS里面转下相应格式再传入，因为相机的色彩空间可能不同。

Imitates the color tracking function of pixel cake. The parameter intensity is the intensity of color migration. 
The larger the face protection parameter value, the closer it is to the original image.
Supports regional color migration, just add SAM semantic segmentation to create a mask。
You can turn on the options to adjust the automatic brightness, contrast, and saturation. 
The parameters for each value are within the automatic adjustment range of 0.5, which is 50% automatic adjustment.
Added a tone switch that can mimic the tone of the original image.
It should also be noted that if it is a camera's original JPG or PNG image, 
it needs to be converted to PS format before being transmitted, as the color space of the camera may be different.

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/imitation1.png?raw=true)

2: IC-Light的自定义光源，可以定义光源各种形状，位置偏移(位置可调整为负数)，缩放，亮度，旋转，背景和光源颜色，图片高斯模糊,
可以多个光源组合在一起，绘制任意图像的光源。

IC-Light's custom light source can define various shapes, position offsets (position can be adjusted to negative numbers), 
scale, brightness, rotation, background and light color，gaussian blur of image.multiple light sources can be combined together to draw the light source of any image.

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/light_source.png?raw=true)

3：图片水印功能，可以添加图片水印和文字水印，可以调整水印的位置，大小，透明度，文字水印还可以自定义字体颜色和字体，可以自己增加字体，
放在插件目录的fonts文件夹下。

The image watermark function can add image watermarks and text watermarks. You can adjust the position, size, 
and transparency of the watermark. The text watermark can also customize the font color and font. You can add fonts yourself and put them in the fonts folder of the plug-in directory.

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/watermark.png?raw=true)

4: AI去水印功能，基于开源的lama模型开发，在水印地方涂抹上蒙板即可去水印

模型下载地址：[模型下载](https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors) 下载模型放到models/ming文件夹

AI watermark removal function, developed based on the open source lama model, can remove the watermark by applying a mask to the watermark area.

Model download address: [model download](https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors) download the model and put it in the models/ming folder

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/remove_watermark.png?raw=true)

5: HSL调色, 调整范围-30到30

HSL color adjustment, adjustment range -30 to 30

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/HSL_color.png?raw=true)

6: 色彩平衡，包含中间调，高光，阴影的调整，调整范围-100到100

Color balance, including midtones, highlights, and shadows adjustments.Adjustment range: -100 to 100

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/color_balance.png?raw=true)

7：阴影高光调节,调整范围-10到10

Shadow highlight adjustment, adjustment range -10 to 10

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/highlight_shadow.png?raw=true)

8：一个是百度翻译，可以中英繁之间的翻译，注意要填入百度的AppID和Appkey，可以到百度翻译申请，免费的。

Baidu Translate, which can translate between Chinese, English and Traditional Chinese. 
Please note that you need to fill in Baidu's AppID and Appkey. You can apply for it at Baidu Translate, it is free.

注册地址 Registered Address：

<https://fanyi-api.baidu.com/?fr=pcHeader&ext_channel=DuSearch>

![Image text](https://github.com/mingsky-ai/ComfyUI-MingNodes/blob/main/images/baidu_translate.png?raw=true)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mingsky-ai/ComfyUI-MingNodes&type=Date)](https://star-history.com/#mingsky-ai/ComfyUI-MingNodes&Date)







