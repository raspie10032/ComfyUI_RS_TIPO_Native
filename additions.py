"""Small, source-grounded additions for the STRING-facing prompt node.

This vocabulary recognizes a limited set of English action equivalents. It is
deliberately not a relationship planner or a general semantic classifier.
"""
import re

from .grounding import key, supported_hair_action
from .scene_context import action_clauses


_ACTIONS = (
    ('give', 'giving', 'offer', 'offering', 'pass', 'passing', 'handing'),
    ('receive', 'receiving', 'accept', 'accepting', 'take', 'taking'),
    ('reach', 'reaching', 'extend', 'extending', 'outstretched'),
    ('hold', 'holding', 'grasp', 'grasping'),
    ('dance', 'dancing'), ('lead', 'leading'), ('follow', 'following'),
    ('hug', 'hugging', 'embrace', 'embracing'),
    ('look', 'looking', 'gaze', 'gazing'),
    ('smile', 'smiling', 'happy'), ('laugh', 'laughing'),
    ('sit', 'sitting', 'seated'), ('stand', 'standing'),
    ('kneel', 'kneeling'), ('lie', 'lying'),
    ('walk', 'walking'), ('run', 'running'),
    ('lean', 'leaning'), ('rest', 'resting'), ('touch', 'touching'),
    ('cross', 'crossed'), ('clasp', 'clasped', 'clasping'),
    ('interlock', 'interlocked', 'interlocking'),
    ('braid', 'braiding', 'braided'), ('brush', 'brushing', 'brushed'),
    ('comb', 'combing', 'combed'),
)
_NOUNS = (
    ('arm', 'arms'), ('hand', 'hands'), ('leg', 'legs'), ('foot', 'feet'),
    ('eye', 'eyes'), ('cup', 'cups', 'teacup'),
)
_ALIASES = {word: group[0] for group in (*_ACTIONS, *_NOUNS) for word in group}
_GRAMMAR = set('a an the at to of in on with and each other another partner partners '
               'person their his her both own together being s'.split())
_ACTION_WORDS = {group[0] for group in _ACTIONS}
_OPPOSITES = ({'give', 'receive'}, {'lead', 'follow'}, {'sit', 'stand', 'kneel', 'lie'})


def concepts(text):
    return {_ALIASES.get(word, word) for word in re.findall(r"[a-z0-9]+", key(text))
            if word not in _GRAMMAR}


def select_additions(original, tags, *, common='', character=True, limit=6):
    """Keep anchored details, then cap additions; never fill a quota.

    A character can receive anchored actions and limited basic posture/gaze
    support. Unmentioned scenery and props cannot enter via a character output.
    Peer roles are intentionally absent from the support set.
    """
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('TIPO max_added_tags must be between 1 and 32.')
    own = concepts(original)
    shared = concepts(common)
    actions = concepts(', '.join((*action_clauses(common), *action_clauses(original))))
    support = actions if character else own
    # Simple aliases such as happy -> smile may occur outside action clauses.
    if character:
        support |= (own | shared) & {'smile', 'laugh'}
    supported_verbs = support & _ACTION_WORDS
    if supported_verbs & {'give', 'receive'}:
        supported_verbs.add('hold')
    # A transfer/reach can need an explicit hand even when the input says arms.
    if supported_verbs & {'give', 'receive', 'reach'}:
        support |= {'hand', 'arm'}
    # Do not erase useful interaction staging just because it is not a literal
    # synonym of the source. Existing protection still rejects explicit conflicts.
    basic = {'sit', 'stand'}
    if not (own | shared) & {'sad', 'angry', 'cry', 'crying'}:
        basic.add('smile')
    partner_action = bool(supported_verbs & {'give', 'receive', 'dance', 'hug', 'touch'})
    accepted, rejected = [], []
    known = set(own)
    for tag in tags:
        words = concepts(tag)
        reason = None
        verbs = words & _ACTION_WORDS
        details = words - _ACTION_WORDS
        partner_gaze = partner_action and key(tag) in {
            'looking at another', 'looking at partner', 'looking at each other'}
        grooming_context = supported_hair_action(original, tag)
        if words and words <= known:
            # Equivalent phrasing does not make a sufficient prompt better.
            reason = 'already expressed by source or accepted additions'
        elif character and not partner_gaze and not grooming_context and (not verbs or not verbs <= supported_verbs | basic or not details <= support):
            reason = 'not grounded in this participant or shared action'
        elif character and any(words & group and support & group and
                               not words & group <= support & group for group in _OPPOSITES):
            reason = 'opposite participant role or posture'
        elif character and not partner_gaze and not details and verbs <= {'hold', 'reach', 'touch', 'look'}:
            reason = 'incomplete action without a supplied object or target'
        elif len(accepted) >= limit:
            reason = 'addition budget reached'
        if reason:
            rejected.append({'tag': tag, 'reason': reason})
        else:
            accepted.append(tag)
            known |= words
    return accepted, rejected
