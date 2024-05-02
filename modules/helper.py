def normalize_suffix(suffix: str) -> str:
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    return suffix
