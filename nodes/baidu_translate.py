import requests
import random
from hashlib import md5


def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


class BaiduTranslateNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "from_translate": (
                    [
                        'auto',
                        'zh',
                        'cht',
                        'en'
                    ],
                    {"default": "auto"},
                ),
                "to_translate": ([
                                     'zh',
                                     'en',
                                     'cht'
                                 ], {"default": "en"}),
                "baidu_appid": ("STRING", {"multiline": False, "placeholder": "Input AppId"}),
                "baidu_appkey": ("STRING", {"multiline": False, "placeholder": "Input AppKey"}),
                "text": ("STRING", {"multiline": True, "placeholder": "Input prompt"}),
            }
        }

    CATEGORY = "MingNode/Translate"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "baidu_translate"

    def baidu_translate(self, from_translate, to_translate, text, baidu_appid, baidu_appkey):
        appid = baidu_appid
        appkey = baidu_appkey
        from_lang = from_translate
        to_lang = to_translate
        endpoint = 'https://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        url = endpoint + path
        query = text
        salt = random.randint(32768, 65536)
        sign = make_md5(appid + query + str(salt) + appkey)

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

        r = requests.post(url, params=payload, headers=headers)
        result = r.json()
        txt = result['trans_result'][0]['dst']

        return (txt,)
