"""Frozen CPU rollout tasks. Optimizers never cross this process boundary."""
from dataclasses import replace
from io import BytesIO
import hashlib
import os
import torch
from texas42.agents import HeuristicAgent
from texas42.encoding import encode
from texas42.selfplay import SelfPlayEnv
from .buffer import Experience
from .model import ActorCritic


def stream_seed(seed, name, index=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}:{index}'.encode()).digest()[:8], 'big') % (2**63)


def snapshot(model):
    out = BytesIO()
    torch.save({key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}, out)
    return out.getvalue()


def from_snapshot(weights, hidden_size):
    with torch.random.fork_rng(devices=[]):
        model = ActorCritic(hidden_size)
    model.load_state_dict(torch.load(BytesIO(weights), map_location='cpu', weights_only=True))
    return model.eval()


@torch.no_grad()
def collect_hand(model, generator, seed, episode):
    """One engine hand, with distinct terminal trajectories for the two learning seats."""
    device = next(model.parameters()).device
    team = (episode + seed) % 2
    env = SelfPlayEnv(episode='hand', seed=stream_seed(seed, 'deal', episode))
    obs = env.reset()
    agents = [HeuristicAgent() for _ in range(4)]
    for seat, agent in enumerate(agents):
        agent.reset(seat)
    paths = {seat: [] for seat in range(4) if seat % 2 == team}
    decisions = 0
    while True:
        if obs.phase == 'play' and obs.seat in paths:
            features = tuple(encode(obs))
            x = torch.tensor([features], dtype=torch.float32, device=device)
            mask = torch.tensor([obs.action_mask], dtype=torch.bool, device=device)
            action, log_prob, value = model.select(x, mask, generator=generator)
            action = int(action.item())
            paths[obs.seat].append(Experience(features, obs.action_mask, action,
                float(log_prob.item()), float(value.item()), 0.0, False, obs.seat, episode))
        else:
            action = agents[obs.seat].act(obs)
        transition = env.step(action)
        decisions += 1
        if transition.terminated:
            break
        obs = transition.observation
    if decisions != 33 or sum(transition.hand_result.team_points) != 42:
        raise RuntimeError('Incomplete normal hand')
    for seat, path in paths.items():
        if len(path) != 7:
            raise RuntimeError('Expected seven decisions per learning seat')
        path[-1] = replace(path[-1], reward=float(transition.rewards[seat]), done=True)
    return episode, tuple(tuple(path) for path in paths.values()), transition.rewards[team]


_WORKER_MODEL = None
_WORKER_VERSION = None


def initialize_worker():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def collect_chunk(task):
    """Load each rollout snapshot once per process, then reuse it for full hands."""
    global _WORKER_MODEL, _WORKER_VERSION
    weights, hidden_size, seed, indices = task
    version = hashlib.sha256(weights).digest()
    if version != _WORKER_VERSION:
        _WORKER_MODEL = from_snapshot(weights, hidden_size)
        _WORKER_VERSION = version
    results = []
    for episode in indices:
        try:
            generator = torch.Generator().manual_seed(stream_seed(seed, 'parallel-episode-v1', episode))
            results.append(collect_hand(_WORKER_MODEL, generator, seed, episode))
        except Exception as exc:
            raise RuntimeError(f'PPO rollout episode={episode} worker_pid={os.getpid()}: {exc}') from exc
    return results, os.getpid()
