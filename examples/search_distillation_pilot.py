"""Offline search-distillation pilot: one regularized improvement step from search.

docs/search_training.md, sections 6.2-6.3. Measures whether training a network
toward model-guided search makes the network itself play better, before
search is built into the training loop. One run is one iteration of the
integrated loop of section 4.5; pass the result back as --source for the next
round (the improved model is then both the policy and the search guide).

Steps (all seeded, and independent of --threads):
  1. Positions. The source checkpoint plays all four seats (greedy: its own
     evaluation distribution) for --games hands, and --positions-per-game
     play decisions with at least two legal moves are sampled per hand
     (examples.search_accuracy.collect_positions). Every
     --validation-every-th hand is held out.
  2. Search. Model-guided PIMC from each position's OWN observation:
     --samples uniform worlds, every legal move rolled out to the end of the
     hand with the source model at all four seats, points objective
     (texas42.belief.evaluate_actions). collect_positions also returns the
     true hands as an accuracy label; they are dropped here and never reach
     the search.
  3. Target. CMPO (research.ppo.search.cmpo_target): pi_bar is proportional to
     pi_old * exp(clip(A / sigma, -c, c)), with A the search advantage and
     sigma its pi_old-weighted RMS over the training states.
  4. Fine-tune. Only the play network, on KL(pi_bar || pi_theta) plus
     --value-weight * (V_theta - V_old)^2 (the trunk is shared; the second
     term keeps the value head's outputs). Adam, early stopping on the
     held-out KL.
  5. Write. OUTPUT_DIR/distilled-c{C}.pt: a version-2 checkpoint identical to
     the source except for the play weights, with metadata['derivation']
     recording the provenance; and OUTPUT_DIR/distilled-c{C}.json with
     diagnostics. The search result is cached in OUTPUT_DIR/searched.pt and
     reused when the settings match, so several --clip values share one search.

Evaluate the results with examples.evaluate_agent_vs_agent (recipes in
docs/command_reference.md).
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

from examples.experiment_output import add_output_arguments, record_output

_MODELS = None  # Worker-local search guide (four seat factories).


def _initialize_search(source):
    """Spawn-worker initializer: single-threaded inference, guide loaded once."""
    global _MODELS
    import torch
    torch.set_num_threads(1)
    from research.ppo.opponents import CheckpointIdentity
    from research.ppo.search import model_factories
    _MODELS = model_factories(CheckpointIdentity.from_path(source))
    _MODELS[0].prepare()


def _search(task):
    """One position: (index, [(action_id, expected points), ...], seconds)."""
    from texas42.belief import evaluate_actions
    index, obs, seed, samples = task
    started = time.perf_counter()
    evaluation = evaluate_actions(obs, rng=random.Random(seed), samples=samples,
                                  rollout_policies=_MODELS, objective='points')
    return index, [(e.action_id, e.expected_points) for e in evaluation.estimates], time.perf_counter() - started


def search_settings(args, identity):
    return dict(source_sha256=identity.sha256, seed=args.seed, games=args.games,
                positions_per_game=args.positions_per_game, samples=args.samples, objective='points',
                belief='uniform', rollouts='source model, greedy, full hand')


def search(args, identity, out):
    """Collect positions and search them; cached in out/searched.pt."""
    import torch
    from texas42.encoding import encode
    from texas42.parallel import OrderedPool, resolve_workers
    from examples.search_accuracy import collect_positions
    from research.ppo.search import model_factories
    from research.ppo.seeds import stream_seed
    settings = search_settings(args, identity)
    cache = out / 'searched.pt'
    if cache.exists():
        data = torch.load(cache, weights_only=True)
        if data.get('settings') == settings:
            print(f'Reusing the search in {cache}')
            return data
        raise SystemExit(f'{cache} holds a search with other settings; use another --output-dir')
    started = time.perf_counter()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Single-threaded inference while collecting positions.
    models = model_factories(identity)
    models[0].prepare()
    positions = collect_positions(models, args.games, args.positions_per_game, args.seed)
    tasks = [(i, p['obs'], stream_seed(args.seed, 'pilot-search', i), args.samples)
             for i, p in enumerate(positions)]  # Observations only: no true hands.
    workers = resolve_workers(args.threads, len(tasks))
    print(f'{len(tasks)} positions from {args.games} hands; searching with {workers} worker(s)', flush=True)
    with OrderedPool(workers, initializer=_initialize_search, initargs=(str(identity.path),)) as pool:
        results = pool.map(_search, tasks)
    torch.set_num_threads(threads)
    width = len(tasks[0][1].action_mask)
    q = torch.zeros(len(tasks), width)
    for index, estimates, _ in results:
        for action_id, value in estimates:
            q[index, action_id] = value
    seconds = [s for _, _, s in results]
    data = dict(settings=settings,
                features=torch.tensor([encode(obs) for _, obs, _, _ in tasks], dtype=torch.float32),
                masks=torch.tensor([obs.action_mask for _, obs, _, _ in tasks], dtype=torch.bool), q=q,
                games=torch.tensor([p['game'] for p in positions]),
                played=torch.tensor([p['action_id'] for p in positions]),
                mean_search_seconds=sum(seconds) / len(seconds), wall_seconds=time.perf_counter() - started)
    torch.save(data, cache)
    return data


def distill(args, identity, data, clip, out):
    """Fine-tune the play network toward the CMPO target; write checkpoint + diagnostics."""
    import torch
    from research.ppo.checkpoint import load_frozen, load_model, read
    from research.ppo.opponents import CheckpointIdentity
    from research.ppo.search import (advantage_scale, cmpo_target, masked_kl, masked_log_softmax,
                                     search_advantages)
    torch.manual_seed(args.seed)
    features, masks, q, games = data['features'], data['masks'], data['q'], data['games']
    validation = torch.tensor([int(g) % args.validation_every == 0 for g in games])
    training = ~validation
    model, _ = load_model(identity.path)
    for network in model.networks.values():
        network.requires_grad_(False)
    net = model.networks['play']
    net.requires_grad_(True)
    old = copy.deepcopy(net).eval()
    with torch.no_grad():
        logits, v_old = old(features)
        log_old = masked_log_softmax(logits, masks)
        advantage, p_old = search_advantages(log_old, q, masks)
        sigma = advantage_scale(p_old[training], advantage[training])
        log_target = cmpo_target(log_old, advantage, masks, clip, sigma)
        step_kl = masked_kl(log_target, log_old, masks)  # KL(pi_bar || pi_old) <= 2c.
        search_best = torch.where(masks, q, torch.full_like(q, -torch.inf)).argmax(-1)
        old_best = log_old.argmax(-1)

    def losses(rows):
        logits, values = net(features[rows])
        kl = masked_kl(log_target[rows], masked_log_softmax(logits, masks[rows]), masks[rows]).mean()
        return kl, ((values - v_old[rows]) ** 2).mean()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    train_rows, val_rows = training.nonzero().squeeze(-1), validation.nonzero().squeeze(-1)
    with torch.no_grad():
        start_train, start_val = float(losses(train_rows)[0]), float(losses(val_rows)[0])
    best_val, best_state, best_epoch, history, stale = start_val, copy.deepcopy(net.state_dict()), 0, [], 0
    for epoch in range(1, args.max_epochs + 1):
        net.train()
        order = train_rows[torch.randperm(len(train_rows), generator=generator)]
        for start in range(0, len(order), args.batch_size):
            kl, drift = losses(order[start:start + args.batch_size])
            optimizer.zero_grad()
            (kl + args.value_weight * drift).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_kl, _ = losses(train_rows)
            val_kl, val_drift = losses(val_rows)
        history.append(dict(epoch=epoch, train_kl=float(train_kl), val_kl=float(val_kl),
                            val_value_rmse=math.sqrt(float(val_drift))))
        if float(val_kl) < best_val - 1e-6:
            best_val, best_state, best_epoch, stale = float(val_kl), copy.deepcopy(net.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        logits, values = net(features)
        log_new = masked_log_softmax(logits, masks)
        new_best = log_new.argmax(-1)

    def share(condition, rows):
        return float(condition[rows].float().mean())

    diagnostics = dict(
        clip=clip, sigma_points=sigma, best_epoch=best_epoch, epochs_run=len(history),
        train_states=int(training.sum()), validation_states=int(validation.sum()),
        target_step_kl_mean=float(step_kl[training].mean()),
        kl_to_target=dict(train_before=start_train,
                          train_after=float(masked_kl(log_target, log_new, masks)[training].mean()),
                          validation_before=start_val, validation_after=best_val),
        search_disagrees_with_policy=dict(train=share(search_best != old_best, training),
                                          validation=share(search_best != old_best, validation)),
        argmax_changed=dict(train=share(new_best != old_best, training),
                            validation=share(new_best != old_best, validation)),
        new_argmax_matches_search=dict(train=share(new_best == search_best, training),
                                       validation=share(new_best == search_best, validation),
                                       validation_before=share(old_best == search_best, validation)),
        kl_new_to_old_validation=float(masked_kl(log_new, log_old, masks)[validation].mean()),
        value_rmse_validation=float(((values - v_old)[validation] ** 2).mean().sqrt()),
        history=history)
    checkpoint = read(identity.path)
    checkpoint['model'] = model.state_dict()
    checkpoint['metadata']['derivation'] = dict(
        kind='search-distillation-pilot-v1', source_path=str(identity.path), source_sha256=identity.sha256,
        phase='play', target='cmpo', clip=float(clip), searched_states=int(training.sum()), seed=args.seed,
        games=args.games, per_game=args.positions_per_game, samples=args.samples, objective='points',
        learning_rate=args.learning_rate, batch=args.batch_size, best_epoch=best_epoch,
        value_weight=args.value_weight)
    path = out / f'distilled-c{clip:g}.pt'
    torch.save(checkpoint, path)
    read(path)  # Validates the version-2 checkpoint.
    result = CheckpointIdentity.from_path(path)
    load_frozen(result)
    diagnostics['checkpoint'] = dict(path=str(path), sha256=result.sha256)
    (out / f'distilled-c{clip:g}.json').write_text(json.dumps(diagnostics, indent=2) + '\n', encoding='utf-8')
    return diagnostics


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True,
                   help='version-2 checkpoint with a trained play network: the policy improved and the search guide')
    p.add_argument('--output-dir', type=Path, required=True, help='search cache, checkpoints and diagnostics')
    p.add_argument('--clip', type=float, nargs='+', default=[2.0], help='CMPO clip c; one checkpoint per value')
    p.add_argument('--games', type=int, default=1000, help='self-play hands to sample positions from')
    p.add_argument('--positions-per-game', type=int, default=4)
    p.add_argument('--samples', type=int, default=16, help='sampled worlds per search')
    p.add_argument('--seed', type=int, default=20261025, help='positions, search streams and fine-tuning')
    p.add_argument('--validation-every', type=int, default=10, help='hold out every N-th hand')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=5, help='epochs without held-out improvement before stopping')
    p.add_argument('--value-weight', type=float, default=1.0, help='weight of the value-preservation term')
    p.add_argument('--threads', type=int, default=0, help='search processes: 0 auto, 1 sequential, N up to N')
    add_output_arguments(p)
    return p


@record_output
def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    for name in ('games', 'positions_per_game', 'samples', 'validation_every', 'batch_size', 'max_epochs',
                 'patience'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.threads < 0 or min(args.clip) <= 0 or args.learning_rate <= 0 or args.value_weight < 0:
        p.error('--threads must be >= 0, --clip and --learning-rate positive, --value-weight >= 0')
    from research.ppo.opponents import CheckpointIdentity
    from research.ppo.search import trained_phases
    identity = CheckpointIdentity.from_path(args.source)
    from research.ppo.checkpoint import load_frozen
    if load_frozen(identity)[1]['checkpoint_version'] != 2 or 'play' not in trained_phases(identity):
        p.error('--source must be a version-2 checkpoint with a trained play network')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = search(args, identity, out)
    if int((data['games'] % args.validation_every == 0).sum()) in (0, len(data['games'])):
        p.error('the validation split is empty or covers every hand; adjust --games or --validation-every')
    print(f"Search: {len(data['q'])} positions, {data['mean_search_seconds']:.3f} s/search")
    rows = []
    for clip in args.clip:
        d = distill(args, identity, data, clip, out)
        print(f"c={clip:g}: best epoch {d['best_epoch']}, held-out KL to target "
              f"{d['kl_to_target']['validation_before']:.4f} -> {d['kl_to_target']['validation_after']:.4f}, "
              f"held-out play choices changed {d['argmax_changed']['validation']:.1%} -> {d['checkpoint']['path']}")
        rows.append({k: v for k, v in d.items() if k != 'history'})
    # record_output flattens a returned list into the CSV export.
    return rows


if __name__ == '__main__':
    main()
