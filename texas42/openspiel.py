"""Optional OpenSpiel adapter; all game rules live in the Texas 42 engine.

Import this module before calling pyspiel.load_game("python_texas42").
"""
from __future__ import annotations

import json

import numpy as np
import pyspiel

from .features import FEATURE_SCHEMA_VERSION, PUBLIC_DECISIONS_PER_HAND, OBSERVER_PHASE_SLOTS
from .episodes import EpisodeState
from .actions import (
    ACTION_REGISTRY, ACTION_SPACE_VERSION, BID_OFFSET, NUM_ACTIONS, PLAY_OFFSET, TRUMP_OFFSET,
)
from .models import trump_name
from .actions import PLAY_ACTIONS
from .rulesets import RULESET_ID, validate_ruleset


# Re-export range boundaries for callers; do not translate engine action IDs.

_GAME_TYPE = pyspiel.GameType(
    short_name="python_texas42",
    long_name="Texas 42 (engine-backed Python adapter)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=4,
    min_num_players=4,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={"episode": "hand", "target_marks": 7, "shaker": -1,
                             "ruleset_id": RULESET_ID},
)


class Texas42Game(pyspiel.Game):
    """One hand (signed mark stakes) or one match (signed unit returns)."""

    action_space_version = ACTION_SPACE_VERSION
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def __init__(self, params=None):
        # Materialize defaults so direct construction and deserialization have
        # identical game strings and tensor capacities.
        params = {"episode": "hand", "target_marks": 7, "shaker": -1,
                  "ruleset_id": RULESET_ID, **(params or {})}
        self.ruleset_id = validate_ruleset(params["ruleset_id"])
        self.episode = params.get("episode", "hand")
        self.target_marks = params.get("target_marks", 7)
        self.shaker = params.get("shaker", -1)
        # Reuse the engine's configuration validation.
        EpisodeState(self.episode, self.target_marks, 0, self.ruleset_id)
        if type(self.shaker) is not int or self.shaker not in range(-1, 4):
            raise ValueError("shaker must be -1 (chance) or a seat from 0 to 3")
        # Every hand awards at least one mark: after 2*T-1 hands some team has T.
        self.max_hands = 1 if self.episode == "hand" else 2 * self.target_marks - 1
        utility = 5.0 if self.episode == "hand" else 1.0
        info = pyspiel.GameInfo(
            num_distinct_actions=NUM_ACTIONS, max_chance_outcomes=28, num_players=4,
            min_utility=-utility, max_utility=utility, utility_sum=0.0,
            max_game_length=PUBLIC_DECISIONS_PER_HAND * self.max_hands,
        )
        super().__init__(_GAME_TYPE, info, params)

    def new_initial_state(self):
        return Texas42State(self)

    def max_chance_nodes_in_history(self):
        # max_game_length counts player decisions; chance has a separate bound.
        return 28 * self.max_hands + int(self.shaker == -1)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return Texas42Observer(
            self, iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params,
        )


class Texas42State(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.session = None if game.shaker == -1 else EpisodeState(
            game.episode, game.target_marks, game.shaker, game.ruleset_id,
        )
        self.deals = [[]]  # Chance actions, grouped into contiguous seven-domino hands.
        self.events = [[]]  # Public (actor, global action) pairs per hand.
        self.shakers = [game.shaker]
        self._returns = [0.0] * 4

    @property
    def hand_index(self):
        return len(self.deals) - 1

    def current_player(self):
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self.session is None or self.session.needs_deal:
            return pyspiel.PlayerId.CHANCE
        return self.session.hand.seat

    def is_terminal(self):
        return self.session is not None and self.session.terminated

    def _legal_actions(self, player):
        if self.is_terminal() or player != self.current_player():
            return []
        if self.is_chance_node():
            return [action for action, _ in self.chance_outcomes()]
        hand = self.session.hand
        return list(hand.legal_action_ids())

    def chance_outcomes(self):
        if not self.is_chance_node():
            return []
        outcomes = list(range(4)) if self.session is None else sorted(
            set(range(28)) - set(self.deals[-1]),
        )
        return [(action, 1.0 / len(outcomes)) for action in outcomes]

    def _apply_action(self, action):
        if self.is_terminal():
            raise ValueError("Cannot act after episode termination")
        if self.is_chance_node():
            if action not in dict(self.chance_outcomes()):
                raise ValueError(f"Illegal chance action: {action}")
            if self.session is None:
                game = self.get_game()
                self.session = EpisodeState(game.episode, game.target_marks, action, game.ruleset_id)
                self.shakers[-1] = action
            else:
                self.deals[-1].append(action)
                if len(self.deals[-1]) == 28:
                    hands = [[PLAY_ACTIONS[i] for i in self.deals[-1][seat * 7:(seat + 1) * 7]]
                             for seat in range(4)]
                    self.session.deal(hands)
            return
        hand = self.session.hand
        actor = hand.seat
        # Delegate validation before appending public history: an illegal move
        # must change neither the engine state nor the adapter's event stream.
        transition = self.session.apply(action)
        self.events[-1].append((actor, action))
        if transition.terminated:
            self._returns = list(map(float, transition.rewards))
        elif self.session.needs_deal:
            self.deals.append([])
            self.events.append([])
            self.shakers.append(self.session.shaker)

    def returns(self):
        return list(self._returns)

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"Shaker P{action}" if self.session is None else f"Deal {PLAY_ACTIONS[action]}"
        entry = ACTION_REGISTRY[action]
        if not entry.implemented:
            return f"Reserved {entry.key}"
        if entry.phase == "play":
            return f"Play {entry.value}"
        if entry.phase == "bid":
            return "Pass" if entry.value is None else f"Bid {entry.value}"
        return f"Trump {trump_name(entry.value)}"

    def __str__(self):
        # Debug string is full simulator state, never a policy observation.
        return json.dumps({"deals": self.deals, "shakers": self.shakers,
                           "events": self.events, "returns": self._returns}, sort_keys=True)


class Texas42Observer:
    """Fixed-size float32 tensor with named views and player-specific privacy.

    Domino sets use 28-bit masks. Public decisions use 33 ordered slots per
    hand, each containing an actor one-hot and a global-action one-hot.
    Perfect recall retains each hand's private set and public events;
    ordinary observations contain only the current hand and its public events.
    """

    def __init__(self, game, obs_type, params=None):
        if params:
            raise ValueError("Observer parameters are not supported")
        self.obs_type = obs_type
        self.num_hands = game.max_hands if obs_type.perfect_recall else 1
        private_players = (4 if obs_type.private_info == pyspiel.PrivateInfoType.ALL_PLAYERS
                           else 1 if obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER
                           else 0)
        pieces = [("player", (4,))]
        if private_players:
            pieces.extend([("hand", (private_players, 28)),
                           ("private_deals", (self.num_hands, private_players, 28))])
        if obs_type.public_info:
            pieces.extend([
                ("actor", (6,)), ("phase", (OBSERVER_PHASE_SLOTS,)), ("hand_index", (1,)),
                ("marks", (2,)), ("points", (2,)), ("deal_counts", (4,)),
                ("shakers", (self.num_hands, 4)),
                ("history", (self.num_hands, PUBLIC_DECISIONS_PER_HAND, 4 + NUM_ACTIONS)),
            ])
        size = sum(int(np.prod(shape)) for _, shape in pieces)
        self.tensor = np.zeros(size, dtype=np.float32)
        self.dict = {}
        offset = 0
        for name, shape in pieces:
            length = int(np.prod(shape))
            # Named arrays share the flat tensor; set_from updates both at once.
            self.dict[name] = self.tensor[offset:offset + length].reshape(shape)
            offset += length

    def set_from(self, state, player):
        if player not in range(4):
            raise ValueError("player must be a seat from 0 to 3")
        self.tensor.fill(0)
        self.dict["player"][player] = 1
        indexes = range(len(state.deals)) if self.obs_type.perfect_recall else [state.hand_index]
        players = (range(4) if self.obs_type.private_info == pyspiel.PrivateInfoType.ALL_PLAYERS
                   else [player])
        if "hand" in self.dict:
            for private_seat, seat in enumerate(players):
                hand = state.session.hand if state.session and not state.session.needs_deal else None
                ids = ([domino.id for domino in hand.hands[seat]] if hand else
                       [])
                self.dict["hand"][private_seat, ids] = 1
                for slot, index in enumerate(indexes):
                    # Chance draws are simulator bookkeeping. Reveal the entire
                    # private set only when all 28 draws finish, so no observer
                    # ever learns an order that perfect recall would need to keep.
                    if len(state.deals[index]) == 28:
                        ids = state.deals[index][seat * 7:(seat + 1) * 7]
                        self.dict["private_deals"][slot, private_seat, ids] = 1
        if "history" in self.dict:
            actor = state.current_player()
            # OpenSpiel uses negative sentinel IDs; tensor slots 4/5 represent
            # chance/terminal and must not be addressed with negative indexing.
            self.dict["actor"][actor if actor >= 0 else 4 if state.is_chance_node() else 5] = 1
            hand = state.session.hand if state.session and not state.session.needs_deal else None
            phase = (4 if state.is_terminal() else 0 if state.is_chance_node()
                     else {"bid": 1, "trump": 2, "play": 3}[hand.phase])
            self.dict["phase"][phase] = 1
            self.dict["hand_index"][0] = state.hand_index
            self.dict["marks"][:] = state.session.marks if state.session else [0, 0]
            self.dict["points"][:] = hand.points if hand else [0, 0]
            for seat in range(4):
                self.dict["deal_counts"][seat] = len([])
            for slot, index in enumerate(indexes):
                shaker = state.shakers[index]
                if shaker >= 0:
                    self.dict["shakers"][slot, shaker] = 1
                for turn, (seat, action) in enumerate(state.events[index]):
                    self.dict["history"][slot, turn, seat] = 1
                    self.dict["history"][slot, turn, 4 + action] = 1

    def string_from(self, state, player):
        self.set_from(state, player)
        # Sparse strings preserve the same information as the tensor, including
        # recall and privacy options; hidden chance history is deliberately absent.
        return json.dumps({name: [(int(i), int(value.flat[i])) for i in np.flatnonzero(value)]
                           for name, value in self.dict.items()},
                          separators=(",", ":"), sort_keys=True)


pyspiel.register_game(_GAME_TYPE, Texas42Game)
