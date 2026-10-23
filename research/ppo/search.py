"""Model-guided Monte Carlo search: a trained checkpoint inside texas42.belief's PIMC.

Plain PIMC (texas42.agents.pimc) samples hidden deals and rolls each candidate
move out with Heuristic V1 for every seat. Its estimates are only as good as
that model of the other players. This module supplies the three ways a
trained network can stand in for the heuristic. Each is a separate option of
the 'pimc' composition strategy (research.ppo.composition):

1. Rollout policy (policy evaluation). The network plays the simulated
   continuations, so an action's estimate answers "what happens if I play this
   and everyone then plays like the model?". This is a Monte Carlo estimate of
   Q_pi(s, a) for the model's policy pi: a one-step policy-improvement
   operator, as in Tesauro & Galperin's rollouts (NIPS 1996) and in the
   recursive Monte Carlo search of Furtak & Buro (2013).

2. Likelihood model (Bayesian inference over deals). The network's legal
   softmax gives P(action | that player's view in a hypothetical deal), which
   weights deals by how well they explain the observed bids and plays.
   This is Policy-Based Inference (Rebstock et al. 2019). A stochastic learned
   policy degenerates the weights less than the deterministic heuristic does
   (see "Weight collapse" in docs/belief.md).

3. Value leaf (truncated search). Rollouts stop after the next trick, and the
   network's value head estimates the rest of the hand. This is the AlphaZero
   pattern of replacing long rollouts with a learned evaluation; it cuts
   rollout length several-fold, and it is what makes search affordable inside
   training (docs/search_training.md). The value head predicts the acting
   team's final signed marks from that player's own observation (an
   information-set value), so the search objective becomes 'marks'.

Privacy: everything here runs on hypothetical worlds built from the observer's
public information (texas42.belief). Building another seat's observation in a
hypothetical world is legitimate: it is a guess, not the real hand.

Data formats: checkpoints are CheckpointIdentity (path + SHA256), loaded once
per process through checkpoint.load_frozen. Factories follow the arena
convention factory(seat, seed) -> Agent and are picklable for spawn workers.
"""
from dataclasses import dataclass

from texas42.selfplay import observation_from_view

from .composition import CompositeAgentFactory, CompositePolicySpec, PhasePolicySpec, PHASES, phase_network
from .opponents import CheckpointIdentity


def trained_phases(identity: CheckpointIdentity) -> tuple[str, ...]:
    """Phases whose network the checkpoint actually trained (version 1 = play only)."""
    from .checkpoint import load_frozen
    _, metadata = load_frozen(identity)
    if metadata['checkpoint_version'] == 1:
        return ('play',)
    return tuple(p for p in PHASES if p in metadata['neural_phases'] and p in metadata['trained_phases'])


def model_policy_spec(identity: CheckpointIdentity, name: str = 'model') -> CompositePolicySpec:
    """The checkpoint as a full player: its trained phases neural, the rest Heuristic V1.

    Example: the shared bid+play study model plays neural bid/play and
    heuristic trump, exactly as it did during training and evaluation.
    """
    trained = trained_phases(identity)
    if not trained:
        raise ValueError(f'{identity.path} has no trained neural phase to guide search')
    return CompositePolicySpec(name, *(PhasePolicySpec('neural', identity) if p in trained
                                       else PhasePolicySpec('heuristic') for p in PHASES))


def model_factories(identity: CheckpointIdentity, *, greedy: bool = True) -> tuple:
    """Four seat factories playing the model, for rollouts and/or likelihoods.

    greedy=True: rollouts follow each player's most likely move (low variance).
    greedy=False: rollouts sample the model's legal softmax with seeded,
    seat-local streams (the model's full behaviour, more variance).
    Likelihood weighting always uses the full softmax (action_probabilities),
    whichever mode the rollouts use.
    """
    factory = CompositeAgentFactory(model_policy_spec(identity), greedy)
    return (factory,) * 4


# Process-local network cache for value leaves: (identity, phase) -> network.
# load_frozen already caches the checkpoint; this avoids repeating the
# trained-phase checks on every leaf evaluation.
_VALUE_NETWORKS = {}


@dataclass(frozen=True)
class ValueLeaf:
    """Leaf evaluator f(state, observer) -> the observer team's expected final signed marks.

    Steps, at the state where a truncated rollout stopped:
      1. Build the acting seat's ordinary Observation in this hypothetical world
         (selfplay.observation_from_view). This is exactly the input the network
         saw during training.
      2. Run that phase's network and read its value head: the PPO return
         estimate, i.e. expected final signed marks for the acting seat's team.
      3. Return it unchanged if the actor is on the observer's team, negated
         otherwise (the hand is zero-sum between the partnerships).
    The phase must be one the checkpoint trained. A value head for an untrained
    phase was never fitted. With until='trick', every stop happens in the play
    phase (or the hand has ended, and then no leaf is needed).
    """

    identity: CheckpointIdentity

    def __call__(self, state, observer: int) -> float:
        import torch
        from texas42.encoding import encode
        obs = observation_from_view(state.view(), ruleset_id=state.ruleset_id)
        key = (self.identity, str(obs.phase))
        if key not in _VALUE_NETWORKS:
            if obs.phase not in trained_phases(self.identity):
                raise ValueError(f'Value leaf needs a trained {obs.phase} network in {self.identity.path}')
            _VALUE_NETWORKS[key] = phase_network(PhasePolicySpec('neural', self.identity), obs.phase)
        with torch.inference_mode():
            _, value = _VALUE_NETWORKS[key](torch.tensor([encode(obs)], dtype=torch.float32))
        value = float(value.item())
        return value if obs.seat % 2 == observer % 2 else -value


__all__ = ['ValueLeaf', 'model_factories', 'model_policy_spec', 'trained_phases']
