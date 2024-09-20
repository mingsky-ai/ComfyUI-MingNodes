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
                "text": ("STRING", {"multiline": True, "placeholder": "Input prompt"}),
            }
        }

    CATEGORY = "MingNode/translate"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "baidu_translate"

    def baidu_translate(self, from_translate, to_translate, text):
        appid = '20240921002156132'
        appkey = 'pGMJz9iNpUJ_n10xbpYs'
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
        print("=========", r)
        result = r.json()
        txt = result['trans_result'][0]['dst']
        print(txt)

        return (txt,)
