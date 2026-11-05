"""Conservative framing repairs for model JSON, without rewriting its text.

Only one document is accepted. Duplicate keys and concatenated documents are
ambiguous for character ownership and must go through the caller's retry path.
"""
import json
import re


class DuplicateKeyError(json.JSONDecodeError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f'Duplicate JSON key: {key!r}', '', 0)
        result[key] = value
    return result


def _repair_syntax(text):
    """Escape unknown backslashes and remove trailing commas outside strings."""
    output, quoted, i, changes = [], False, 0, []
    while i < len(text):
        char = text[i]
        if quoted and char == '\\':
            following = text[i+1:i+2]
            unicode_escape = following == 'u' and re.fullmatch(r'[0-9a-fA-F]{4}', text[i+2:i+6])
            if following and (following in '"\\/bfnrt' or unicode_escape):
                count = 6 if unicode_escape else 2
                output.append(text[i:i+count])
                i += count
                continue
            output.append('\\\\')
            changes.append('invalid_json_escape')
        elif char == '"':
            quoted = not quoted
            output.append(char)
        elif char == ',' and not quoted and text[i+1:].lstrip().startswith(('}', ']')):
            changes.append('trailing_json_comma')
        else:
            output.append(char)
        i += 1
    return ''.join(output), changes


def decode_document(text, repairs=None):
    text = text.strip().lstrip('\ufeff')
    changes = []
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, count=1, flags=re.I)
        text = re.sub(r'\s*```\s*$', '', text, count=1).strip()
        changes.append('json_code_fence')
    decoder = json.JSONDecoder(object_pairs_hook=unique_object, strict=False)
    try:
        value, end = decoder.raw_decode(text)
    except DuplicateKeyError:
        raise
    except json.JSONDecodeError:
        text, repaired = _repair_syntax(text)
        changes.extend(repaired)
        value, end = decoder.raw_decode(text)
    tail = text[end:].strip()
    if tail:
        # A full object/array followed only by excess closing delimiters has a
        # unique boundary. Do not discard prose, another object or array items.
        if not isinstance(value, (dict, list)) or not re.fullmatch(r'[}\]\s]+', tail):
            raise json.JSONDecodeError('Extra data after JSON document', text, end)
        changes.append('extra_closing_delimiters')
    if repairs is not None:
        repairs.extend(dict.fromkeys(changes))
    return value


def scene_json(text, repairs=None):
    """Decode a scene; retain the older, narrow malformed-common repair."""
    text = text.strip()
    try:
        return decode_document(text, repairs)
    except DuplicateKeyError:
        raise
    except json.JSONDecodeError as original:
        # Only repair the common field if its following character map is fully
        # decodable. Quotes/backslashes inside participant text stay untouched.
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I).strip()
        match = re.fullmatch(r'\{\s*"common"\s*:\s*"(?P<body>.*?)'
            r'"characters"\s*:\s*(?P<characters>\{.*)\}\s*', clean, re.S)
        if match is None:
            raise original
        try:
            changes = []
            characters = decode_document(match['characters'], changes)
        except (json.JSONDecodeError, ValueError):
            raise original from None
        if (not isinstance(characters, dict) or re.search(r'(?<!\\)"[^"\r\n]*"\s*:', match['body'])):
            raise original
        body = match['body'].rstrip()
        if body.endswith(','):
            body = body[:-1].rstrip()
        if body.endswith('"'):
            body = body[:-1]
        elif not re.search(r'(?<!\\)"', body):
            raise original
        # JSON's string decoder unescapes valid sequences; repair only unknown
        # escapes and literal control characters. Interior quotes remain text.
        escaped = re.sub(r'(?<!\\)"', r'\\"', body)
        try:
            common = decode_document('"' + escaped + '"', changes)
        except (json.JSONDecodeError, ValueError):
            raise original from None
        if repairs is not None:
            repairs.extend(dict.fromkeys((*changes, 'common_string_delimiters')))
        return {'common': common, 'characters': characters}
