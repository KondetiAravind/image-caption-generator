import requests

LANGUAGES = {
    "Hindi": "hi", "French": "fr", "Spanish": "es",
    "German": "de", "Tamil": "ta", "Telugu": "te",
    "Japanese": "ja", "Arabic": "ar",
}

def translate_caption(text, target_lang):
    try:
        url    = "https://translate.googleapis.com/translate_a/single"
        params = {"client":"gtx","sl":"en","tl":target_lang,"dt":"t","q":text}
        resp   = requests.get(url, params=params, timeout=5)
        return resp.json()[0][0][0]
    except:
        return "[Translation unavailable]"

def translate_to_all(caption, languages=None):
    if languages is None:
        languages = LANGUAGES
    out = {"English": caption}
    for lang, code in languages.items():
        out[lang] = translate_caption(caption, code)
    return out
