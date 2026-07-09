def truncate_text(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + '...'
    return text

def slugify(text: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9]+', '_', text).lower()