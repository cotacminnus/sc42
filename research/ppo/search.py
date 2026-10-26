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
import math
import random

from texas42.selfplay import observation_from_view

from .composition import (CompositeAgentFactory, CompositePolicySpec, PairedAgentFactory, PhasePolicySpec,
                          PHASES, _NeuralPolicy, partner_index, phase_network)
from .opponents import CheckpointIdentity


def trained_phases(identity: CheckpointIdentity) -> tuple[str, ...]:
    """Phases whose network the checkpoint actually trained (version 1 = play only)."""
    from .checkpoint import load_frozen
    _, metadata = load_frozen(identity)
    if metadata['checkpoint_version'] == 1:
        return ('play',)
    return tuple(p for p in PHASES if p in metadata['neural_phases'] and p in metadata['trained_phases'])


def value_heads_trained(identity: CheckpointIdentity) -> bool:
    """Whether the checkpoint's local value heads were fitted as PPO's baseline.

    Only the local actor-critic trains them. Centralized-critic training uses
    the privileged critic in their place in the value loss, so they keep their
    initial weights. Policy-only training (critic_mode 'none') never fits them.
    Neither may score search leaves. Checkpoints that predate critic modes
    trained the local heads.
    """
    from .checkpoint import load_frozen
    _, metadata = load_frozen(identity)
    return (metadata.get('critic_mode') or 'local') == 'local'


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


def model_factories(identity: CheckpointIdentity, *, greedy: bool = True,
                    partner: CheckpointIdentity | None = None) -> tuple:
    """Four seat factories playing the model, for rollouts and/or likelihoods.

    greedy=True: rollouts follow each player's most likely move (low variance).
    greedy=False: rollouts sample the model's legal softmax with seeded,
    seat-local streams (the model's full behaviour, more variance).
    Likelihood weighting always uses the full softmax (action_probabilities),
    whichever mode the rollouts use.

    partner: a paired-partner run's second network. Each partnership's first
    seat (0/1) then plays `identity` and its second seat (2/3) plays `partner`,
    the same seat rule used in paired training and evaluation
    (composition.partner_index). Both teams are modelled as the pair.
    """
    factory = CompositeAgentFactory(model_policy_spec(identity), greedy)
    if partner is None:
        return (factory,) * 4
    second = CompositeAgentFactory(model_policy_spec(partner, 'model partner'), greedy)
    return (PairedAgentFactory(factory, second),) * 4


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
    The phase must be one the checkpoint trained, and the checkpoint must have
    trained its local value heads (value_heads_trained): an unfitted head
    would silently score every leaf with its initial weights. With
    until='trick' or 'turn', every stop happens in the play phase (or the hand
    has ended, and then no leaf is needed).

    partner (optional, paired-partner runs): the acting seat's own network
    answers, i.e. `identity` for a team's first seat (0/1) and `partner` for its
    second seat (2/3), matching model_factories(partner=...).
    """

    identity: CheckpointIdentity
    partner: CheckpointIdentity | None = None

    def __call__(self, state, observer: int) -> float:
        import torch
        from texas42.encoding import encode
        obs = observation_from_view(state.view(), ruleset_id=state.ruleset_id)
        identity = (self.identity if self.partner is None or partner_index(obs.seat) == 0
                    else self.partner)
        key = (identity, str(obs.phase))
        if key not in _VALUE_NETWORKS:
            if obs.phase not in trained_phases(identity):
                raise ValueError(f'Value leaf needs a trained {obs.phase} network in {identity.path}')
            if not value_heads_trained(identity):
                raise ValueError(f'Value leaf needs trained local value heads; {identity.path} was trained '
                                 'with a centralized or no critic')
            _VALUE_NETWORKS[key] = phase_network(PhasePolicySpec('neural', identity), obs.phase)
        with torch.inference_mode():
            _, value = _VALUE_NETWORKS[key](torch.tensor([encode(obs)], dtype=torch.float32))
        value = float(value.item())
        return value if obs.seat % 2 == observer % 2 else -value


# ---------------------------------------------------------------------------
# Search inside training (docs/search_training.md, sections 8 and 10).
#   P1: InMemoryGuide plays the learner's CURRENT network inside the search,
#       from the rollout snapshot a collection worker already holds (the
#       learner has no checkpoint file mid-rollout).
#   P3: TrainingSearch + search_values: the collector searches a seeded
#       fraction of learner play decisions and stores every legal move's
#       estimate beside the decision (buffer.Experience.search_q).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingSearch:
    """Collector settings for searching learner play decisions (P3).

    fraction  probability that a learner play decision is searched. Drawn from
              a dedicated per-hand stream, one draw per learner play decision,
              so which decisions are searched never depends on worker
              scheduling. Forced moves (one legal action) are never searched.
    samples   sampled worlds per search: uniform belief from the learner's own
              observation, full-hand rollouts, points objective. Those settings
              pass the move-ranking gate (section 6); value leaves do not.
    """
    fraction: float
    samples: int

    def __post_init__(self):
        if not (math.isfinite(self.fraction) and 0 < self.fraction <= 1):
            raise ValueError('search fraction must be in (0, 1]')
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError('search samples must be a positive integer')


class _GuideAgent:
    """Greedy player for an in-memory PhaseActorCritic (P1).

    Each neural phase takes the masked argmax of its network's logits (the
    composition fast path, _NeuralPolicy); every other phase plays Heuristic
    V1. That is how the learner plays greedily during collection
    (agent.PPOAgent), and how model_factories plays a checkpoint. No RNG is
    consumed.
    """

    def __init__(self, model):
        from texas42.agents import HeuristicAgent
        self.delegates = {p: (_NeuralPolicy(model.networks[p], 0, True) if p in model.neural_phases
                              else HeuristicAgent()) for p in PHASES}

    def reset(self, seat):
        for delegate in self.delegates.values():
            delegate.reset(seat)

    def act(self, obs):
        return self.delegates[obs.phase].act(obs)


@dataclass(frozen=True, eq=False)
class InMemoryGuide:
    """Arena-style seat factory, factory(seat, seed) -> Agent, for a network in memory."""
    model: object

    def __call__(self, seat, seed):
        return _GuideAgent(self.model)


def search_values(obs, guides, samples, seed):
    """Full-rollout search estimate (points) of every legal move, as a tuple over all action IDs.

    guides are four seat factories playing the current policy (InMemoryGuide);
    entries for illegal actions are 0.0. Worlds come from the observation only,
    never from the real hidden hands.
    """
    from texas42.belief import evaluate_actions
    evaluation = evaluate_actions(obs, rng=random.Random(seed), samples=samples, rollout_policies=guides,
                                  objective='points')
    values = [0.0] * len(obs.action_mask)
    for estimate in evaluation.estimates:
        values[estimate.action_id] = estimate.expected_points
    return tuple(values)


# ---------------------------------------------------------------------------
# Search-derived policy targets (docs/search_training.md, sections 4.3-4.4).
# Shared by the offline distillation pilot (examples/search_distillation_pilot)
# and the search loss in training. Batched tensors, one row per searched
# state, 58 action columns; masks are the legal-action masks.
# ---------------------------------------------------------------------------

def masked_log_softmax(logits, masks):
    """Log-probabilities over legal moves only (-inf on illegal ones)."""
    import torch
    return torch.log_softmax(logits.masked_fill(~masks, -torch.inf), -1)


def search_advantages(log_old, q, masks):
    """(Â, π_old): Â(a) = Q̂(a) - Σ_b π_old(b) Q̂(b) on legal moves, 0 elsewhere.

    log_old is the masked log-policy that the search improves on, and q holds
    the search's estimate for each legal move (other entries are ignored). The
    baseline Σ π_old Q̂ makes Â average to zero under π_old, which leaves the
    policy gradient unchanged at θ_old (section 4.3).
    """
    import torch
    zeros = torch.zeros_like(q)
    p_old = torch.where(masks, log_old.exp(), zeros)
    baseline = (p_old * torch.where(masks, q, zeros)).sum(-1, keepdim=True)
    return torch.where(masks, q - baseline, zeros), p_old


def advantage_scale(p_old, advantage):
    """σ = sqrt(mean over states of Σ_a π_old(a) Â(a)²), the CMPO normalizer."""
    return float((p_old * advantage ** 2).sum(-1).mean().sqrt())


def cmpo_target(log_old, advantage, masks, clip, sigma):
    """Muesli's clipped MPO target: log π̄ with π̄ ∝ π_old · exp(clip(Â/σ, -c, c)).

    The tilt keeps every legal move (π̄ > 0 wherever π_old > 0), changes each
    probability by at most a factor e^(2c), and bounds KL(π̄ ‖ π_old) ≤ 2c
    (section 4.4).
    """
    x = (advantage / sigma).clamp(-clip, clip)
    return masked_log_softmax(log_old + x, masks)


def masked_kl(log_p, log_q, masks):
    """KL(p ‖ q) per row over legal moves (illegal entries are -inf in both)."""
    import torch
    return torch.where(masks, log_p.exp() * (log_p - log_q), torch.zeros_like(log_p)).sum(-1)


# ---------------------------------------------------------------------------
# P4: the search loss inside the PPO update (docs/search_training.md, 4.3-4.4, 8).
# ---------------------------------------------------------------------------

SEARCH_TARGETS = ('cmpo', 'advantage')


def search_update_targets(model, data, target, clip):
    """Fix the update's search targets at θ_old; add them to `data`; return diagnostics.

    Run once per update, before the first optimizer step, so π_old is the
    rollout policy. Rows where data['searched'] is set carry search_q. Added
    full-length tensors (meaningful only on searched rows):
      search_log_old     log π_old over legal moves;
      search_advantage   Â / σ: search advantage, scaled by its π_old-weighted
                         RMS over the update's searched states (as CMPO does);
      search_log_target  log π̄, the CMPO target ('cmpo' only).
    """
    import torch
    rows = data['searched']
    masks = data['masks'][rows]
    with torch.no_grad():
        logits, _ = model(data['features'][rows])
        log_old = masked_log_softmax(logits, masks)
        advantage, p_old = search_advantages(log_old, data['search_q'][rows], masks)
        sigma = advantage_scale(p_old, advantage) or 1.0  # All moves equal: no signal at any scale.
        q = torch.where(masks, data['search_q'][rows], torch.full_like(log_old, -torch.inf))

        def full(values):
            out = torch.zeros_like(data['search_q'])
            out[rows] = values
            return out

        data['search_log_old'] = full(log_old)
        data['search_advantage'] = full(advantage / sigma)
        diagnostics = dict(searched_states=int(rows.sum()), search_advantage_scale=sigma,
                           search_disagreement=float((q.argmax(-1) != log_old.argmax(-1)).float().mean()),
                           search_abs_advantage=float((p_old * advantage.abs()).sum(-1).mean()))
        if target == 'cmpo':
            log_target = cmpo_target(log_old, advantage, masks, clip, sigma)
            data['search_log_target'] = full(log_target)
            diagnostics['search_target_kl'] = float(masked_kl(log_target, log_old, masks).mean())
    return diagnostics


def search_minibatch_loss(model, batch, target, clip_epsilon):
    """(sum of per-state search losses, searched rows) for one minibatch, or None.

    cmpo       KL(π̄ ‖ π_θ): fit the regularized improvement target (4.4).
    advantage  -Σ_a π_old(a) min(r(a) Â(a), clip(r(a), 1±ε) Â(a)), with
               r = π_θ/π_old: PPO's clipped surrogate summed over every legal
               move, whose gradient at θ_old is the all-action gradient
               Σ_a ∇π_θ(a) Â(a) (4.3).
    """
    import torch
    rows = batch['searched']
    count = int(rows.sum())
    if not count:
        return None
    masks = batch['masks'][rows]
    logits, _ = model(batch['features'][rows])
    log_new = masked_log_softmax(logits, masks)
    if target == 'cmpo':
        losses = masked_kl(batch['search_log_target'][rows], log_new, masks)
    else:
        log_old = batch['search_log_old'][rows]
        # Fill illegal entries before exp: -inf - -inf would be NaN, even in a gradient.
        ratio = (log_new - log_old).masked_fill(~masks, 0.0).exp()
        p_old = log_old.exp().masked_fill(~masks, 0.0)
        advantage = batch['search_advantage'][rows]
        surrogate = torch.minimum(ratio * advantage,
                                  ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
        losses = -(p_old * surrogate).sum(-1)
    return losses.sum(), count


__all__ = ['InMemoryGuide', 'SEARCH_TARGETS', 'TrainingSearch', 'ValueLeaf', 'advantage_scale', 'cmpo_target',
           'masked_kl', 'masked_log_softmax', 'model_factories', 'model_policy_spec', 'search_advantages',
           'search_minibatch_loss', 'search_update_targets', 'search_values', 'trained_phases',
           'value_heads_trained']
