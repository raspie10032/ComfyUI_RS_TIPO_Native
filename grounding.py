"""Conservative lexical guard for additive TIPO tags; not a semantic planner.

Source strings remain byte-for-byte authoritative. Private character attributes
never enter the scene's allow-list. Unknown relationships still require review.
"""
import re

# Strip explicit weight wrappers only in model inputs/comparison keys. Keep
# escaped parentheses, artist: prefixes and names containing parentheses intact.
_WEIGHT = re.compile(r'(?<!\\)\(([^()]*?)(?<!\\):\s*[+-]?\d+(?:\.\d+)?\s*\)')
_V4 = re.compile(r'(?<!\S)[+-]?\d+(?:\.\d+)?::(.*?)::', re.S)


def model_text(text):
    old = None
    while old != text:
        old, text = text, _WEIGHT.sub(r'\1', text)
    return _V4.sub(r'\1', text)


def key(text):
    return model_text(text).strip().casefold().replace('_', ' ')


# These additions change identity, wardrobe, composition, era or rendering. In
# protected mode TIPO can enrich actions/scene, but cannot invent these fields.
_PROTECTED = re.compile(
    r'\b(?:hair|eyes?|eyebrows?|eyelashes|skin|ears?|horns?|tail|wings?|'
    r'breasts?|cleavage|collarbone|navel|thighs?|armpits?|abs|muscles?|nose|'
    r'bangs|bob cut|ponytail|twintails?|braids?|bald|makeup|lipstick|nails?|nail polish|'
    r'girl|girls|boy|boys|female|male|woman|women|man|men|child|children|'
    r'adolescent|mature|young|old|asian|hetero|yuri|yaoi|couple|'
    r'shirt|dress|skirt|sweater|cardigan|jacket|coat|sleeves?|sleeveless|'
    r'gloves?|boots?|shoes?|socks?|stockings|pants|shorts|uniform|ribbon|'
    r'earrings?|necklace|jewelry|bracelet|bangle|turtleneck|lingerie|'
    r'underwear|bare|strap|straps|clothes|clothing|outfit|'
    r'style|artstyle|realistic|photorealistic|photograph|3d|render|cgi|'
    r'artist|signature|watermark|dated|copyright|text|lettering|'
    r'solo|focus|pov|portrait|profile|shot|close-up|upper body|full body|out of frame|'
    r'simple background|white background)\b|\b\d+(?:girls?|boys?)\b|\b(?:19|20)\d\d\b')
_SCENE_PEOPLE = re.compile(
    r'\b(?:holding|hug|hugging|embrace|dancing|hand|hands|arm|arms|leg|legs|'
    r'head|face|mouth|lips|blush|smile|smiling|laughing|crying|looking|gaze|'
    r'sitting|standing|kneeling|lying|reaching|offering|receiving|leaning|'
    r'kissing|walking|running|jumping|touching|clasped|interlocked|fingers?)\b')
_VIEWS = re.compile(r'\b(?:from behind|from above|from below|from side|looking back|looking up|looking down|looking to the side|looking ahead)\b')
_HAND_TARGET = re.compile(r"\b(?:hand|hands) on (?:another(?:'s)?|other|partner(?:'s)?) (waist|shoulder|back|head|arm|face)\b")


def guard_additions(original, tags, *, common='', interaction='', character=True):
    """Return accepted/rejected additions, with observable rejection reasons.

    Unspecified attributes are protected too. This intentionally sacrifices
    decoration to avoid changing a participant's appearance or role.
    """
    own = key(original)
    context = key(' '.join((common, interaction, original)))
    explicit = {key(t) for t in original.split(',') if key(t)}
    hand_targets = set(_HAND_TARGET.findall(own))
    accepted, rejected = [], []
    known = set(explicit)
    for tag in tags:
        tag = tag.strip()
        value = key(tag)
        if not value or value in known:
            continue
        # A canonical tag already inside a weighted source is redundant.
        if re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', own):
            continue
        reason = None
        if _PROTECTED.search(value):
            reason = 'unrequested identity, wardrobe, style or composition'
        elif not character and _SCENE_PEOPLE.search(value):
            reason = 'participant-specific addition in shared scene'
        elif _VIEWS.search(value) and value not in context:
            reason = 'unrequested viewpoint or interaction orientation'
        elif 'from behind' in value and 'from behind' not in context:
            reason = 'unrequested interaction orientation'
        elif re.search(r'\b(?:hand|hands|arm|arms) on own\b', value) and re.search(
                r'\b(?:other|another|partner|hug\w*|offer\w*|receiv\w*|reach\w*|danc\w*)\b', own):
            reason = 'self-directed gesture conflicts with supplied partner action'
        elif 'looking at viewer' in value and re.search(
                r'look\w* (?:at )?(?:each other|one another|another|other|partner)', context):
            reason = 'viewer gaze conflicts with partner gaze'
        elif any(re.search(r'\b(?:' + a + r')\b', own) and re.search(r'\b(?:' + b + r')\b', value)
                 for a, b in [('standing', 'sitting|kneeling|lying'),
                              ('sitting', 'standing|kneeling|lying'),
                              ('lying', 'standing|sitting|kneeling'),
                              ('offering', 'receiving'), ('receiving', 'offering'),
                              ('offering|receiving', 'drinking|eating|kissing|hugging|dancing'),
                              ('leading', 'following'), ('following', 'leading')]):
            reason = 'action conflicts with supplied role'
        elif hand_targets and _HAND_TARGET.search(value) and not hand_targets.intersection(_HAND_TARGET.findall(value)):
            reason = 'partner contact target conflicts with supplied role'
        elif ('open mouth' in known and value == 'closed mouth') or (
                'closed mouth' in known and value in ('open mouth', 'parted lips')) or (
                'parted lips' in known and value == 'closed mouth'):
            reason = 'conflicting expression'
        if reason:
            rejected.append({'tag': tag, 'reason': reason})
        else:
            accepted.append(tag)
            known.add(value)
    return accepted, rejected
