"""Rules identity is separate from action identity and model output layout."""
from .models import ContractKind

RULESET_ID = "cadets_base_v1"


def validate_ruleset(ruleset_id: str) -> str:
    # Never silently load a checkpoint under different house rules. Additional
    # identifiers belong here only once their engine rules are implemented
    # (for example a Nello ruleset; see docs/roadmap.md step 10).
    if type(ruleset_id) is not str or ruleset_id != RULESET_ID:
        raise ValueError(f"Unsupported ruleset: {ruleset_id!r}")
    return ruleset_id


def enabled_contracts(ruleset_id: str = RULESET_ID) -> tuple[ContractKind, ...]:
    validate_ruleset(ruleset_id)
    return (ContractKind.NORMAL,)
