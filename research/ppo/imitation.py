"""Separate teacher initialization and fixed anchor-play forgetting diagnostics."""
import torch
from texas42.agents import HeuristicAgent
from texas42.encoding import encode
from texas42.selfplay import SelfPlayEnv
from .agent import PPOAgent
from .seeds import stream_seed


@torch.no_grad()
def teacher_data(seed, hands, *, anchor=None):
    features, masks, actions, forced = [], [], [], []
    for episode in range(hands):
        domain = 'bid-bc-deals-v1' if anchor is None else 'anchor-kl-deals-v1'
        env = SelfPlayEnv(seed=stream_seed(seed, domain, episode))
        obs = env.reset()
        agents = [HeuristicAgent() if anchor is None else PPOAgent(anchor) for _ in range(4)]
        for seat, agent in enumerate(agents):
            agent.reset(seat)
        while True:
            action = agents[obs.seat].act(obs)
            desired = 'bid' if anchor is None else 'play'
            if obs.phase == desired:
                features.append(encode(obs)); masks.append(obs.action_mask); actions.append(action)
                forced.append(bool(obs.view.forced) if desired == 'bid' else False)
            transition = env.step(action)
            if transition.terminated:
                break
            obs = transition.observation
    return dict(features=torch.tensor(features, dtype=torch.float32), masks=torch.tensor(masks, dtype=torch.bool),
                actions=torch.tensor(actions), forced=torch.tensor(forced, dtype=torch.bool))


def clone_bidding(trainer, hands, epochs=4):
    if 'bid' not in trainer.config.learn_phases:
        raise ValueError('Behavior cloning requires trainable bidding')
    data = teacher_data(trainer.config.seed, hands)
    generator = torch.Generator().manual_seed(stream_seed(trainer.config.seed, 'bid-bc-shuffle-v1'))
    model = trainer.model.networks['bid']
    # Fresh temporary optimizer: PPO starts with a new Adam state after imitation.
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer.config.learning_rate)
    for _ in range(epochs):
        order = torch.randperm(len(data['actions']), generator=generator)
        for indices in order.split(trainer.config.minibatch_size):
            x, masks, actions = (data[key][indices].to(trainer.device) for key in ('features', 'masks', 'actions'))
            logp, _, _ = model.evaluate(x, masks, actions)
            loss = -logp.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite imitation loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
    with torch.no_grad():
        x, masks, actions = (data[k].to(trainer.device) for k in ('features', 'masks', 'actions'))
        distribution, _ = model.distribution(x, masks)
        correct = distribution.logits.argmax(-1) == actions
        forced = data['forced'].to(trainer.device)
        def accuracy(selected):
            return float(correct[selected].float().mean()) if selected.any() else None
        metrics = dict(hands=hands, samples=len(actions), loss=float(-distribution.log_prob(actions).mean()),
                       accuracy=float(correct.float().mean()), forced_accuracy=accuracy(forced),
                       voluntary_accuracy=accuracy(~forced), forced_samples=int(forced.sum()),
                       voluntary_samples=int((~forced).sum()), epochs=epochs)
    trainer.model.trained_phases = tuple(p for p in ('bid', 'trump', 'play')
        if p in set(getattr(trainer.model, 'trained_phases', ('play',))) | {'bid'})
    trainer.bc_metrics = metrics
    return metrics


@torch.no_grad()
def playing_kl(model, anchor, data):
    device = next(model.parameters()).device
    features, masks = data['features'].to(device), data['masks'].to(device)
    current, _ = model.networks['play'].distribution(features, masks)
    reference, _ = anchor.distribution(data['features'], data['masks'])
    if reference.logits.device != device:
        reference = torch.distributions.Categorical(logits=reference.logits.to(device))
    return float(torch.distributions.kl_divergence(reference, current).mean())
