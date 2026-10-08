"""Pure global-action legality shared by engine state and player adapters."""
from .actions import ACTION_REGISTRY
from .models import Bid
from .players import BidView, PlayView, TrumpView
from .rules import is_legal_bid, legal_plays


def legal_action_ids(view: BidView | TrumpView | PlayView) -> tuple[int, ...]:
    if isinstance(view, BidView):
        return tuple(entry.action_id for entry in ACTION_REGISTRY.implemented_for_phase('bid')
                     if (not view.forced if entry.value is None
                         else isinstance(entry.value, Bid)
                         and (view.current_bid is None or isinstance(view.current_bid, Bid))
                         and is_legal_bid(entry.value, view.current_bid)))
    if isinstance(view, TrumpView):
        return tuple(entry.action_id for entry in ACTION_REGISTRY.implemented_for_phase('trump'))
    if isinstance(view, PlayView):
        # Derive from hand/lead/trump, not a potentially stale legal_plays field.
        return tuple(sorted(d.id for d in legal_plays(
            view.hand, view.trick[0][1] if view.trick else None, view.trump)))
    raise TypeError('Expected a BidView, TrumpView or PlayView')
