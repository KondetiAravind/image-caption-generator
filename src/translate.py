from googletrans import Translator

LANGUAGES = {
    "Hindi": "hi", "French": "fr", "Spanish": "es",
    "German": "de", "Tamil": "ta", "Telugu": "te",
    "Japanese": "ja", "Arabic": "ar",
}

_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator

def translate_to_all(caption, languages=None):
    if languages is None:
        languages = LANGUAGES
    out = {"English": caption}
    t   = get_translator()
    for lang, code in languages.items():
        try:
            out[lang] = t.translate(caption, src="en", dest=code).text
        except Exception as e:
            out[lang] = f"[Error: {e}]"
    return out
