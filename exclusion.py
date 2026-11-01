"""Request-local, decoded-tag constraints used *before* accepting a next token.

No unwanted-tag text is inserted into the positive TIPO input. Comparing decoded
candidates also covers tokenizations other than tokenizer.encode(bad_tag).
"""
import re
import unicodedata

from .grounding import model_text


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', model_text(text)).casefold()
                    .replace('_', ' ').split())


def parse_excluded(text):
    if not isinstance(text, str):
        raise ValueError('TIPO excluded_tags must be a STRING. / 제외 태그는 문자열입니다.')
    tags = tuple(dict.fromkeys(normalize(t) for t in re.split(r'[,\n\r]+', text) if normalize(t)))
    if len(tags) > 256 or len(text) > 16384:
        raise ValueError('TIPO: use at most 256 excluded tags and 16384 characters.')
    if any('<|' in tag or '|>' in tag for tag in tags):
        raise ValueError('TIPO: exclude ordinary tags, not model format/control tokens.')
    return tags


def validate_sources(tags, sources):
    """Source text remains authoritative; resolve contradictions before loading."""
    wanted = set(tags)
    for label, text in sources:
        for part in re.split(r'[,\n\r]+', model_text(text)):
            tag = normalize(part)
            if tag in wanted:
                raise ValueError(f'TIPO excluded_tags conflicts with {label}: {tag}. '
                                 'Remove it from the source or exclusion list. / '
                                 '원문과 제외 목록이 충돌합니다. 한쪽을 수정해 주세요.')


class TagExclusion:
    def __init__(self, tokenizer, tags):
        self.tokenizer = tokenizer
        self.tags = frozenset(tags)
        self.rejections = 0
        self.hits = {}

    def allows(self, generated, candidate):
        # The source contains its own trailing separator. Only generated text is
        # tested: source weights, attention ownership and repetition history stay
        # unchanged. Decode full history for context-dependent BPE decoders.
        text = self.tokenizer.decode([*generated, candidate], skip_special_tokens=False)
        fields = re.sub(r'<\|[^|]*\|>', '\n', text)
        for part in re.split(r'[,\n\r]', fields):
            if part.lstrip().startswith('tag:'):
                part = part.lstrip()[4:]
            tag = normalize(part)
            if tag in self.tags:
                self.rejections += 1
                self.hits[tag] = self.hits.get(tag, 0) + 1
                return False
        return True

    def report(self):
        return {'method': 'decoded-tag-next-token-block', 'tags': sorted(self.tags),
                'blocked_candidates': self.rejections, 'hits': dict(self.hits)}
