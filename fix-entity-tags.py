"""
SAFE FIXER - Run this in your project folder.
It ONLY removes the bad bracket tags like [entity-...] and keeps all your code.
It makes a .bak backup first, so nothing is lost.
Usage: python fix_entity_tags.py
"""
import re
import pathlib

for folder in ["src", "scripts"]:
    root = pathlib.Path(folder)
    if not root.exists():
        continue
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        original = text
        # Backup
        if "【" in text or "】" in text:
            p.with_suffix(".py.bak").write_text(original, encoding="utf-8")
            # Pattern 1: [entity-XXX;canonical_name=YYY] -> YYY
            text = re.sub(r"【entity-([^;¦\]]+);[^】]*】", r"\1", text)
            # Pattern 2: any remaining [....] -> remove brackets only, keep inner text before ; if present
            def _clean(m):
                inner = m.group(1)
                if ";" in inner:
                    inner = inner.split(";")[0]
                if "¦" in inner:
                    inner = inner.split("¦")[0]
                inner = inner.replace("entity-", "")
                return inner
            text = re.sub(r"【([^】]*)】", _clean, text)
            # Remove stray bracket chars if any left
            text = text.replace("【", "").replace("】", "")
            p.write_text(text, encoding="utf-8")
            print(f"Fixed: {p} (backup saved as {p}.py.bak)")
        else:
            print(f"OK (no bad chars): {p}")
print("Done! Now run: python -m src.method1_bicubic")
