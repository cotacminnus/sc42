"""Estimation accuracy and cost of PIMC search guided by heuristics vs a trained model.

Question: when a trained model replaces Heuristic V1 inside Monte Carlo search,
do the search's hidden-deal BELIEFS and its VALUE ESTIMATES get closer to the
truth, and what does each search mode cost? (docs/belief.md, "Model-guided
PIMC"; the timings feed docs/search_training.md.)

Procedure
  1. Play complete hands with a fixed "table": the trained model at all four
     seats (default), or Heuristic V1 (--table heuristic). Each hand's deal and
     seats come from --seed, so runs are reproducible.
  2. From each hand, sample --positions-per-game play decisions with at least
     two legal moves. For each, record the acting seat's ordinary Observation,
     the move the table actually made, and, as an evaluation-only label, the
     true remaining hands (SelfPlayEnv.remaining_hands, the documented
     privileged channel for label runners). The label is never passed to search.
  3. After the hand, record the realized signed marks for that seat's team.
  4. For each position, score:
     Beliefs: how much probability each belief puts on the TRUE owner of every
       unseen domino (mean probability; log-loss with 1% uniform smoothing so a
       sampled belief that missed the truth stays finite), plus effective
       sample size. Modes: exact uniform marginals; heuristic-likelihood belief;
       model-likelihood belief.
     Values: PIMC's estimate of signed marks for the move actually played vs
       the realized signed marks (bias, RMSE, correlation), and whether the
       search's best move equals the table's move. Modes: heuristic rollouts;
       model rollouts; model rollouts + value leaf (rollouts stop after one trick).
       All modes use objective='marks' and the same rollout seeds per position
       (common random numbers), so differences reflect the mode.
     Timing: wall-clock seconds per search, per mode.

Reading the results: realized marks are ONE noisy sample of each position's
true value, so RMSE includes irreducible outcome noise; compare modes with each
other, not against zero. Bias and correlation are the more informative columns.
With the model table, model rollouts simulate the true continuation policy;
with the heuristic table, heuristic rollouts do. Running both tables separates
"better model of the players" from "better search".
"""
import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from time import perf_counter

from examples.experiment_output import add_output_arguments, record_output
from research.ppo.opponents import CheckpointIdentity
from research.ppo.search import ValueLeaf, model_factories
from research.ppo.seeds import stream_seed
from texas42.agents import variant_factory
from texas42.belief import build_belief, evaluate_actions, hidden_constraints, tile_marginals
from texas42.selfplay import SelfPlayEnv

BELIEF_MODES = ('uniform', 'heuristic-likelihood', 'model-likelihood')
VALUE_MODES = ('heuristic-rollouts', 'model-rollouts', 'model-rollouts+value-leaf')
SMOOTHING = 0.01  # Uniform mixture weight in the belief log-loss.


def collect_positions(table, games, per_game, seed):
    """Play `games` hands with the table policy; return sampled labelled positions.

    Each position: dict(obs, action_id, true_hands, game, realized_marks).
    true_hands are absolute-seat tuples of domino IDs (privileged label only).
    """
    positions = []
    for game in range(games):
        env = SelfPlayEnv(seed=stream_seed(seed, 'accuracy-deal', game))
        obs = env.reset()
        agents = [factory(seat, stream_seed(seed, 'accuracy-agent', game * 4 + seat))
                  for seat, factory in enumerate(table)]
        for seat, agent in enumerate(agents):
            agent.reset(seat)
        candidates, result = [], None
        while obs is not None:
            action = agents[obs.seat].act(obs)
            if obs.phase == 'play' and len(obs.legal_action_ids) > 1:
                candidates.append(dict(obs=obs, action_id=action, true_hands=env.remaining_hands(), game=game))
            step = env.step(action)
            result = step.hand_result or result
            obs = step.observation
        picker = random.Random(stream_seed(seed, 'accuracy-positions', game))
        for position in picker.sample(candidates, min(per_game, len(candidates))):
            team = position['obs'].seat % 2
            position['realized_marks'] = result.marks_awarded * (1 if result.scoring_team == team else -1)
            positions.append(position)
    return positions


def belief_scores(position, factories, samples, seed):
    """Per-mode (mean P(true owner), smoothed log-loss, ESS) over the unseen dominoes."""
    obs, true_hands = position['obs'], position['true_hands']
    owner = {d: seat for seat, hand in enumerate(true_hands) for d in hand}
    constraints = hidden_constraints(obs)
    uniform = tile_marginals(constraints)  # Exact, also the smoothing reference.

    def score(marginals, ess):
        probs = [marginals[d][owner[d]] for d in constraints.unknown]
        smoothed = [(1 - SMOOTHING) * p + SMOOTHING * uniform[d][owner[d]]
                    for p, d in zip(probs, constraints.unknown)]
        return dict(p_true=mean(probs), log_loss=mean(-math.log(p) for p in smoothed), ess=ess)

    scores = {'uniform': score(uniform, None)}
    for mode, policies in (('heuristic-likelihood', factories['heuristic']),
                           ('model-likelihood', factories['model'])):
        belief = build_belief(obs, rng=random.Random(seed), samples=samples, likelihood_policies=policies)
        scores[mode] = score(belief.marginals(), belief.effective_sample_size)
    return scores


def value_scores(position, factories, leaf, samples, seed):
    """Per-mode (estimate for the played move, best move, seconds); same seeds for all modes."""
    obs, results = position['obs'], {}
    settings = {
        'heuristic-rollouts': dict(rollout_policies=factories['heuristic']),
        'model-rollouts': dict(rollout_policies=factories['model']),
        'model-rollouts+value-leaf': dict(rollout_policies=factories['model'], until='trick', leaf_evaluator=leaf),
    }
    for mode, options in settings.items():
        started = perf_counter()
        evaluation = evaluate_actions(obs, rng=random.Random(seed), samples=samples, objective='marks', **options)
        seconds = perf_counter() - started
        estimate = next(e.expected_points for e in evaluation.estimates if e.action_id == position['action_id'])
        results[mode] = dict(estimate=estimate, best=evaluation.best_action_id, seconds=seconds)
    return results


def correlation(xs, ys):
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if sx == 0 or sy == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def summarize(rows):
    beliefs = {mode: dict(p_true=mean(r['beliefs'][mode]['p_true'] for r in rows),
                          log_loss=mean(r['beliefs'][mode]['log_loss'] for r in rows),
                          mean_ess=None if mode == 'uniform' else mean(r['beliefs'][mode]['ess'] for r in rows))
               for mode in BELIEF_MODES}
    realized = [r['realized_marks'] for r in rows]
    values = {}
    for mode in VALUE_MODES:
        estimates = [r['values'][mode]['estimate'] for r in rows]
        errors = [e - y for e, y in zip(estimates, realized)]
        values[mode] = dict(bias=mean(errors), rmse=math.sqrt(mean(e * e for e in errors)),
                            correlation=correlation(estimates, realized),
                            agreement=mean(r['values'][mode]['best'] == r['action_id'] for r in rows),
                            mean_seconds=mean(r['values'][mode]['seconds'] for r in rows),
                            max_seconds=max(r['values'][mode]['seconds'] for r in rows))
    return dict(positions=len(rows), beliefs=beliefs, values=values)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True, help='trained checkpoint guiding search')
    p.add_argument('--table', choices=('model', 'heuristic'), default='model',
                   help='policy playing all four seats in the evaluated hands')
    p.add_argument('--games', type=int, default=50)
    p.add_argument('--positions-per-game', type=int, default=2)
    p.add_argument('--samples', type=int, default=16, help='sampled deals per belief/search')
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--output', type=Path, help='JSON results (rows + summary)')
    add_output_arguments(p)
    return p


@record_output
def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    for name in ('games', 'positions_per_game', 'samples'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    import torch
    torch.set_num_threads(1)  # Single-threaded inference: comparable per-search timings.
    identity = CheckpointIdentity.from_path(args.model)
    factories = dict(heuristic=(variant_factory('HHH'),) * 4, model=model_factories(identity))
    factories['model'][0].prepare()
    leaf = ValueLeaf(identity)
    table = factories['model'] if args.table == 'model' else factories['heuristic']
    print(f'Model {identity.path} [{identity.sha256}]; table: {args.table}; {args.games} games x '
          f'{args.positions_per_game} positions; {args.samples} samples; seed {args.seed}')
    positions = collect_positions(table, args.games, args.positions_per_game, args.seed)
    rows = []
    for index, position in enumerate(positions):
        seed = stream_seed(args.seed, 'accuracy-search', index)
        rows.append(dict(game=position['game'], seat=position['obs'].seat, action_id=position['action_id'],
                         tricks_completed=len(position['obs'].view.completed_tricks),
                         realized_marks=position['realized_marks'],
                         beliefs=belief_scores(position, factories, args.samples, seed),
                         values=value_scores(position, factories, leaf, args.samples, seed)))
    summary = summarize(rows)
    print(f'\n{summary["positions"]} positions')
    print('\nBeliefs (higher P(true owner) and lower log-loss are better)')
    print(f'{"mode":24s} {"P(true owner)":>14s} {"log-loss":>9s} {"mean ESS":>9s}')
    for mode, s in summary['beliefs'].items():
        ess = '-' if s['mean_ess'] is None else f'{s["mean_ess"]:.1f}'
        print(f'{mode:24s} {s["p_true"]:14.3f} {s["log_loss"]:9.3f} {ess:>9s}')
    print('\nValue estimates for the played move vs realized signed marks')
    print(f'{"mode":26s} {"bias":>7s} {"RMSE":>6s} {"corr":>6s} {"agrees":>7s} {"s/search":>9s} {"max s":>6s}')
    for mode, s in summary['values'].items():
        corr = '-' if s['correlation'] is None else f'{s["correlation"]:.3f}'
        print(f'{mode:26s} {s["bias"]:+7.3f} {s["rmse"]:6.3f} {corr:>6s} {s["agreement"]:7.1%} '
              f'{s["mean_seconds"]:9.3f} {s["max_seconds"]:6.2f}')
    if args.output:
        data = dict(model=dict(path=identity.path, sha256=identity.sha256), table=args.table, games=args.games,
                    positions_per_game=args.positions_per_game, samples=args.samples, seed=args.seed,
                    smoothing=SMOOTHING, summary=summary, rows=rows)
        args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'\nJSON results saved to {args.output}')
    # record_output flattens a returned list into the per-position CSV export.
    return rows


if __name__ == '__main__':
    main()
