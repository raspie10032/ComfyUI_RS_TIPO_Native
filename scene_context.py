"""Derive shared action context from existing inputs, without a second prompt UI.

Only recognizable action/position clauses are shared across owners. Unknown
clauses remain in their owner's tag input, rather than exposing appearance.
This is a conservative projection, not a separate LLM or a role planner.
"""
import re

from .grounding import key, model_text

_ACTION = re.compile(
    r'^(?:(?:on|at|to|from) (?:the )?(?:left|right|center|middle)\b|'
    r'(?:left|right) (?:side|person)\b|'
    r'(?:holding|offering|receiving|accepting|giving|taking|passing|handing|'
    r'extending|reaching|leading|following|dancing|hugging|embracing|leaning|'
    r'resting|touching|grasping|clasping|interlocking|supporting|carrying|'
    r'helping|lifting|pulling|pushing|facing|looking|gazing|smiling|laughing|'
    r'sitting|standing|kneeling|lying|walking|running|jumping|waving|pointing|'
    r'playing|reading|writing|talking|speaking|listening|kissing|sharing|'
    r'drinking|eating|feeding|petting|bowing|crouching|stretching|bending|'
    r'hand|hands|arm|arms|head|face|cheek|forehead|back|foot|feet|leg|legs)\b)')
# A recognized verb does not make the rest of its clause safe to share.
_APPEARANCE = re.compile(
    r'\b(?:hair|eyes?|skin|ears?|horns?|tail|wings?|bangs|ponytail|braids?|'
    r'dress|shirt|skirt|sweater|cardigan|uniform|jacket|coat|outfit|clothes|'
    r'clothing|wearing|blonde|brunette|redhead|artist|style|lora)\b|<[^>]+>')


def action_clauses(prompt):
    """Preserve complete recognized clauses; never fabricate a partner action."""
    result = []
    for clause in re.split(r'[,;\n]+', model_text(prompt)):
        clause = clause.strip(' .')
        normalized = key(clause)
        if _ACTION.search(normalized) and not _APPEARANCE.search(normalized):
            if normalized not in {key(item) for item in result}:
                result.append(clause)
    return tuple(result)


def scene_context(common, characters, target=None):
    """Caption context for one independent output, built only from user inputs.

    The focus person's role is repeated last to anchor the requested completion.
    Labels identify input slots, not a claimed image layout. Actual positions
    come only from explicit user clauses; no left-to-right order is invented.
    """
    roles = tuple(action_clauses(prompt) for prompt in characters)
    parts = [model_text(common).strip()] if common.strip() else []
    for index, clauses in enumerate(roles):
        if clauses:
            parts.append(f"Person {index + 1}: " + ', '.join(clauses) + '.')
    if target is not None and roles[target]:
        parts.append('The person being described: ' + ', '.join(roles[target]) + '.')
    return ' '.join(parts), roles
