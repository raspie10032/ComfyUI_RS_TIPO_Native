"""Keep explicit image-prompt weights outside the language model's contract."""
from dataclasses import dataclass
import re

_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
_SD_SUFFIX = re.compile(r':\s*' + _NUMBER + r'\s*$')
_NAI = re.compile(r'(?<![\w.:\\])' + _NUMBER + r'::(.*?)::', re.S)
_KOREAN = re.compile('[\uac00-\ud7af]')


def _escaped(text, index):
    start = index
    while start and text[start - 1] == '\\':
        start -= 1
    return (index - start) % 2 == 1


def _core(text):
    return text.strip(' \t\r\n,;')


def _translate(text, translations):
    core = _core(text)
    if core and core in translations:
        start = text.index(core)
        return text[:start] + translations[core] + text[start + len(core):]
    return text


@dataclass(frozen=True)
class WeightGroup:
    prefix: str
    body: 'PromptWeights'
    suffix: str

    def render(self, translations):
        return self.prefix + self.body.render(translations) + self.suffix


@dataclass(frozen=True)
class PromptWeights:
    parts: tuple

    @property
    def groups(self):
        return tuple(part for part in self.parts if isinstance(part, WeightGroup))

    def plain(self, translations=None):
        translations = translations or {}
        return ''.join(part.body.plain(translations) if isinstance(part, WeightGroup)
                       else _translate(part, translations) for part in self.parts)

    def render(self, translations=None):
        translations = translations or {}
        return ''.join(part.render(translations) if isinstance(part, WeightGroup)
                       else _translate(part, translations) for part in self.parts)

    def translation_terms(self, inside=False):
        """Translate only Korean text inside weights; numbers never enter a request."""
        terms = []
        for part in self.parts:
            if isinstance(part, WeightGroup):
                terms.extend(part.body.translation_terms(inside=True))
            elif inside and _KOREAN.search(part) and _core(part):
                terms.append(_core(part))
        return tuple(dict.fromkeys(terms))

    def restore(self, output, *, tags, translations=None):
        if not self.groups:
            return output
        translations = translations or {}
        # The model may invent a weight despite seeing none. Only source
        # numeric weights are authoritative; keep all generated descriptions.
        output = detach_weights(output).plain()
        if tags:
            from .expansion import append_completion
            from .gemma_prompt import deduplicate_tag_rendering
            prefix = ', '.join(group.render(translations) for group in self.groups)
            output, _ = append_completion(prefix, ' '.join(output.split()))
            return deduplicate_tag_rendering((), output)
        for group in self.groups:
            phrase = group.body.plain(translations).strip(' ,')
            literal = phrase.replace('\\(', '(').replace('\\)', ')')
            pattern = re.escape(literal).replace(r'\ ', r'[\s_]+').replace(',', r'\s*,\s*')
            match = re.search(r'(?<!\w)' + pattern + r'(?!\w)', output, re.I) if phrase else None
            weighted = group.render(translations)
            if match:
                output = output[:match.start()] + weighted + output[match.end():]
            else:
                # A paraphrase/missing source phrase must not erase the rest
                # of the expanded field or guess which new words to weight.
                output = output.rstrip() + (' ' if output.strip() else '') + weighted
        return output


def detach_weights(text, _depth=0):
    """Detach explicit SD/NAI groups, preserving nesting and escaped name brackets.

    Bare (emphasis) and [deemphasis] are left alone. This function owns numeric
    source weights, not every possible prompt-parser dialect.
    """
    if _depth > 64:
        raise ValueError('Image-prompt weights are nested too deeply (maximum 64).')
    spans, stack = [], []
    for index, char in enumerate(text):
        if char not in '()' or _escaped(text, index):
            continue
        if char == '(':
            stack.append(index)
        elif stack:
            start = stack.pop()
            suffix = _SD_SUFFIX.search(text, start + 1, index)
            if suffix and not _escaped(text, suffix.start()):
                spans.append((start, index + 1, start + 1, suffix.start()))
    for match in _NAI.finditer(text):
        spans.append((match.start(), match.end(), match.start(1), match.end(1)))
    parts, end = [], 0
    for start, stop, body_start, body_stop in sorted(spans, key=lambda s: (s[0], -s[1])):
        if start < end:
            continue
        parts.append(text[end:start])
        parts.append(WeightGroup(text[start:body_start],
            detach_weights(text[body_start:body_stop], _depth + 1), text[body_stop:stop]))
        end = stop
    parts.append(text[end:])
    return PromptWeights(tuple(parts))
