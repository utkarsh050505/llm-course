from helper_functions import build_pretokenizer_pattern
import regex as re

pattern = build_pretokenizer_pattern(["<|endoftext|>"])
print(pattern.pattern)
text = "Hello <|endoftext|> world"

matches = list(pattern.finditer(text))
for m in matches:
    print(f"Match: '{m.group(0)}'")