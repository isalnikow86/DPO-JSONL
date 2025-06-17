def make_boring_version(text: str) -> str:
    """
    Entfernt kindliche Sprache, Emojis und Emotionalität.
    Kürzt auf max. 2 neutrale Sätze.
    """
    import re
    sentences = text.replace('\n', ' ').split('.')
    neutral_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Entferne Ausrufe, Emojis, Kinderwörter
        s = re.sub(r'[!⭐️😊🎉]+', '', s)
        s = re.sub(r'\b(toll|lustig|cool|witzig|spannend|super|klasse|spaß|gern|liebe)\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            neutral_sentences.append(s + ".")
        if len(neutral_sentences) >= 2:
            break
    return " ".join(neutral_sentences) or "Information nicht verfügbar."
