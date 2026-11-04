"""Language detection only. All languages use the same LoRA-patched Gemma."""
import re

# Saved workflows may contain these values; they no longer select a second model.
MODES = ('auto', 'english', 'korean')

def has_korean(text):
    return bool(re.search('[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]', text))
