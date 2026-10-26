"""Belief-based Monte Carlo agent: pick the move with the best expected points.

Strategy (Perfect Information Monte Carlo, PIMC; theory in docs/belief.md):
  1. From the agent's own Observation, build a belief over the hidden deal
     (texas42.belief.build_belief): every consistent deal when there are few,
     otherwise `samples` deals drawn exactly uniformly.
  2. In each hypothetical deal, play every legal move, then roll the hand out
     to the end with Heuristic V1 continuations for all four seats.
  3. Choose the move with the highest belief-weighted expected points for this
     agent's team (texas42.belief.evaluate_actions). Every move is tried on the
     same deals and rollout seeds, so differences reflect the moves, not luck.

Phases: trump and play use PIMC. Bidding delegates to HeuristicAgent, because
expected points is a poor bidding objective: a bid barely changes how many
points are taken, only what the bidding team must reach. The phase-composition
evaluator can still pair PIMC trump/play with any other bidding strategy.

Applicable circumstances: an imperfect-information search baseline needing no
training. It assumes the other players behave like Heuristic V1, and it
inherits PIMC's known weaknesses (strategy fusion, non-locality; see
docs/belief.md). Cost is roughly samples × legal moves × one rollout per
decision, about 0.1–2 s with 32–64 samples, so evaluate on modest deal counts.
"""
from __future__ import annotations

import random

from ..belief import (DEFAULT_ENUMERATE_BELOW, DEFAULT_EPSILON, STOP_CONDITIONS, ActionEvaluation,
                      evaluate_actions)
from ..phases import Phase
from ..selfplay import Observation
from .hybrid import variant_factory
from .simpleheuristic import HeuristicAgent

DEFAULT_PIMC_SAMPLES = 32


class PIMCAgent:
    """Ordinary Agent: reset(seat), act(Observation) -> legal global action ID.

    Randomness: one random.Random(seed) per instance supplies belief sampling
    and rollout seeds, so equal seeds and equal observation sequences give equal
    actions. reset() assigns a seat without rewinding the stream, like
    RandomAgent. Each decision consumes a variable number of draws.

    This agent deliberately does not implement ActionProbabilities. Its
    decisions consume RNG, so it cannot answer the pure probability query used
    for belief weighting.

    `last_evaluation` keeps the most recent texas42.belief.ActionEvaluation (or
    None) for diagnostics: per-move expected points, standard errors, world
    counts. It is informational only and never affects later decisions.

    Model-guided search (optional; all default to the original heuristic agent):
      rollout_policies     four arena factories that play the simulated
                           continuations. None = Heuristic V1 for every seat.
                           A trained model here makes rollouts predict how that
                           model would actually continue (policy evaluation).
      likelihood_policies  four factories whose ActionProbabilities weight the
                           hidden deals by how well they explain the observed
                           bids and plays (texas42.belief.policy_likelihood).
                           None = all consistent deals equally likely.
      epsilon              likelihood softening for unexpected moves.
      until, leaf_evaluator, objective
                           truncated search: stop each rollout early ("trick")
                           and let leaf_evaluator estimate the rest, in the
                           objective's units (see evaluate_actions).
    texas42 stays dependency-free: neural factories and value leaves are
    injected by research.ppo.composition / research.ppo.search.
    """

    def __init__(self, seed: int | None = None, *, samples: int = DEFAULT_PIMC_SAMPLES,
                 enumerate_below: int = DEFAULT_ENUMERATE_BELOW, rollout_policies=None,
                 likelihood_policies=None, epsilon: float = DEFAULT_EPSILON, until: str = 'hand',
                 leaf_evaluator=None, objective: str = 'points'):
        if type(samples) is not int or samples < 1:
            raise ValueError('samples must be a positive integer')
        if type(enumerate_below) is not int or enumerate_below < 0:
            raise ValueError('enumerate_below must be a nonnegative integer')
        self.samples = samples
        self.enumerate_below = enumerate_below
        self._rng = random.Random(seed)
        self._heuristic = HeuristicAgent()
        if until not in STOP_CONDITIONS:
            raise ValueError(f'until must be one of {STOP_CONDITIONS}')
        if objective not in ('points', 'marks'):
            raise ValueError("objective must be 'points' or 'marks'")
        if objective == 'marks' and until != 'hand' and leaf_evaluator is None:
            raise ValueError('Stopping rollouts early with the marks objective needs a leaf_evaluator')
        for policies in (rollout_policies, likelihood_policies):
            if policies is not None and len(policies) != 4:
                raise ValueError('Provide one policy factory per seat')
        # Rollout continuations: picklable arena factories. Default Heuristic V1
        # in all phases and seats. Fresh agents are built inside every rollout.
        self._rollout_policies = tuple(rollout_policies) if rollout_policies else (variant_factory('HHH'),) * 4
        self._likelihood_policies = tuple(likelihood_policies) if likelihood_policies else None
        self.epsilon, self.until, self.objective = epsilon, until, objective
        self._leaf_evaluator = leaf_evaluator
        self.seat: int | None = None
        self.last_evaluation: ActionEvaluation | None = None

    def reset(self, seat: int) -> None:
        if type(seat) is not int or seat not in range(4):
            raise ValueError('seat must be an integer from 0 to 3')
        self.seat = seat
        self._heuristic.reset(seat)
        self.last_evaluation = None

    def act(self, obs: Observation) -> int:
        if obs.seat != self.seat:
            raise RuntimeError('Reset this agent for the acting seat before use')
        if obs.phase == Phase.BID:
            return self._heuristic.act(obs)
        if obs.phase not in (Phase.TRUMP, Phase.PLAY):
            raise ValueError('Unknown decision phase')
        if len(obs.legal_action_ids) == 1:
            # Forced move: nothing to compare, so skip the search (no RNG draws).
            self.last_evaluation = None
            return obs.legal_action_ids[0]
        evaluation = evaluate_actions(obs, rng=self._rng, rollout_policies=self._rollout_policies,
                                      samples=self.samples, enumerate_below=self.enumerate_below,
                                      likelihood_policies=self._likelihood_policies, epsilon=self.epsilon,
                                      until=self.until, leaf_evaluator=self._leaf_evaluator,
                                      objective=self.objective)
        self.last_evaluation = evaluation
        return evaluation.best_action_id

    def would_pass_if_allowed(self, obs: Observation) -> bool:
        """Pure BidPreference diagnostic, answered by the heuristic bidding rule."""
        return self._heuristic.would_pass_if_allowed(obs)
