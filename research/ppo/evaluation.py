"""Frozen-policy duplicate evaluation; no training or RNG-stream updates."""
from texas42.agents.hybrid import variant_factory
from texas42.arena import compare_agents
from .agent import PPOAgent


def evaluate(model, *, opponent='HHH', deals=128, seed=12345, bootstrap_seed=0, bootstrap_samples=200):
    def factory(seat, agent_seed):
        return PPOAgent(model, seed=agent_seed, greedy=True)
    return compare_agents(factory, variant_factory(opponent), num_deals=deals,
                          seed=seed, bootstrap_seed=bootstrap_seed,
                          bootstrap_samples=bootstrap_samples)
