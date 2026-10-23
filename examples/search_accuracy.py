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

Ranking check (--rank-check): a search that is calibrated on average can still
  rank the ALTERNATIVE moves badly, which is what matters for choosing moves and
  for training on search advantages (docs/search_training.md, gate P0). The
  check adds points-objective full rollouts and compares, per position, the
  move ranking of a candidate search mode with a reference mode (same worlds
  and seeds): pairwise order concordance, top-1 agreement, Spearman rank
  correlation, and ranking regret = Q_ref(best by reference) - Q_ref(best by
  candidate), the reference-measured cost of trusting the candidate. A
  replicate of the reference with independent worlds and seeds gives the noise
  scale: how much two estimates of the same quantity disagree at --samples.
  --partner-model makes the guide a paired-partner PAIR (first seats play
  --model, second seats --partner-model; the value leaf uses the acting seat's
  own network).

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
# --rank-check only: full rollouts scored in points (the objective that plays
# best), and the (name, candidate, reference) ranking comparisons:
#   truncation: does stopping after one trick with the value head (marks) keep
#               the ranking of full model rollouts (marks)?
#   objective:  does scoring full rollouts in marks keep the points ranking?
#   total:      the value leaf against the best-playing configuration.
#   guide:      heuristic rollouts against model rollouts (both points).
#   noise:      the reference again with independent worlds and seeds. Not a
#               strict ceiling: the other candidates share the reference's
#               worlds (common random numbers), the replicate does not.
REPLICATE_MODE = 'model-rollouts:points:replicate'
POINTS_MODES = ('heuristic-rollouts:points', 'model-rollouts:points', REPLICATE_MODE)
RANK_PAIRS = (
    ('noise', REPLICATE_MODE, 'model-rollouts:points'),
    ('truncation', 'model-rollouts+value-leaf', 'model-rollouts'),
    ('objective', 'model-rollouts', 'model-rollouts:points'),
    ('total', 'model-rollouts+value-leaf', 'model-rollouts:points'),
    ('guide', 'heuristic-rollouts:points', 'model-rollouts:points'),
)


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


def value_scores(position, factories, leaf, samples, seed, replicate_seed=None):
    """Per-mode search results for one position; the same seed (worlds + rollouts) for every mode.

    Each mode: estimate for the played move, best move and seconds.
    replicate_seed (given only with --rank-check) adds q, the estimate of EVERY
    legal move as [[action_id, value], ...] (a list, which the CSV export
    skips), and the points-objective modes, which are used only for ranking.
    The replicate mode alone uses replicate_seed (independent worlds).
    Marks-objective modes feed the value-accuracy summary. Without
    --rank-check the output is unchanged.
    """
    rank_check = replicate_seed is not None
    obs, results = position['obs'], {}
    settings = {
        'heuristic-rollouts': dict(rollout_policies=factories['heuristic']),
        'model-rollouts': dict(rollout_policies=factories['model']),
        'model-rollouts+value-leaf': dict(rollout_policies=factories['model'], until='trick', leaf_evaluator=leaf),
    }
    if rank_check:
        settings['heuristic-rollouts:points'] = dict(rollout_policies=factories['heuristic'])
        settings['model-rollouts:points'] = dict(rollout_policies=factories['model'])
        settings[REPLICATE_MODE] = dict(rollout_policies=factories['model'])
    for mode, options in settings.items():
        objective = 'points' if ':points' in mode else 'marks'
        rng = random.Random(replicate_seed if mode == REPLICATE_MODE else seed)
        started = perf_counter()
        evaluation = evaluate_actions(obs, rng=rng, samples=samples, objective=objective, **options)
        seconds = perf_counter() - started
        estimate = next(e.expected_points for e in evaluation.estimates if e.action_id == position['action_id'])
        results[mode] = dict(estimate=estimate, best=evaluation.best_action_id, seconds=seconds)
        if rank_check:
            results[mode]['q'] = [[e.action_id, e.expected_points] for e in evaluation.estimates]
    return results


def _ranks(values):
    """Average ranks (1 = lowest), with ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def best_action(q):
    """The move evaluate_actions would choose: highest value, ties to the lowest ID."""
    return max(sorted(q), key=lambda a: (q[a], -a))


def ranking(candidate, reference):
    """Compare two searches' estimates of the same position's legal moves.

    Inputs: dicts {action_id: estimate} over the same legal moves.
    Returns:
      agree, ordered  pairwise concordance counts: among move pairs that BOTH
                      searches order strictly, how many they order the same way
                      (Kendall-style; ties carry no ranking information);
      top1            1.0 if both pick the same best move, else 0.0;
      spearman        rank correlation over the moves, or None with fewer than
                      3 moves or a constant ranking (undefined);
      regret          reference(best by reference) - reference(best by
                      candidate) >= 0: what trusting the candidate costs,
                      measured by the reference.
    """
    actions = sorted(reference)
    agree = ordered = 0
    for i, a in enumerate(actions):
        for b in actions[i + 1:]:
            r, c = reference[a] - reference[b], candidate[a] - candidate[b]
            if r != 0 and c != 0:
                ordered += 1
                agree += (r > 0) == (c > 0)
    spearman = None
    if len(actions) >= 3:
        x, y = _ranks([candidate[a] for a in actions]), _ranks([reference[a] for a in actions])
        spearman = correlation(x, y)
    best_reference, best_candidate = best_action(reference), best_action(candidate)
    return dict(agree=agree, ordered=ordered, top1=float(best_reference == best_candidate),
                spearman=spearman, regret=reference[best_reference] - reference[best_candidate])


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
    summary = dict(positions=len(rows), beliefs=beliefs, values=values)
    if rows and all(mode in rows[0]['values'] for mode in POINTS_MODES):
        summary['ranking'] = {}
        for name, candidate, reference in RANK_PAIRS:
            results = [ranking(dict(r['values'][candidate]['q']), dict(r['values'][reference]['q']))
                       for r in rows]
            ordered = sum(x['ordered'] for x in results)
            spearman = [x['spearman'] for x in results if x['spearman'] is not None]
            summary['ranking'][name] = dict(
                candidate=candidate, reference=reference,
                concordance=sum(x['agree'] for x in results) / ordered if ordered else None,
                top1=mean(x['top1'] for x in results),
                spearman=mean(spearman) if spearman else None, spearman_positions=len(spearman),
                regret=mean(x['regret'] for x in results),
                regret_units='marks' if reference == 'model-rollouts' else 'points')
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True, help='trained checkpoint guiding search')
    p.add_argument('--partner-model', type=Path,
                   help='paired-partner second network: the guide becomes the pair (seats 2/3 play it)')
    p.add_argument('--rank-check', action='store_true',
                   help='also run points-objective full rollouts and compare move rankings between modes')
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
    partner = CheckpointIdentity.from_path(args.partner_model) if args.partner_model else None
    factories = dict(heuristic=(variant_factory('HHH'),) * 4, model=model_factories(identity, partner=partner))
    factories['model'][0].prepare()
    leaf = ValueLeaf(identity, partner)
    table = factories['model'] if args.table == 'model' else factories['heuristic']
    if partner is not None:
        print(f'Partner model (seats 2/3) {partner.path} [{partner.sha256}]')
    print(f'Model {identity.path} [{identity.sha256}]; table: {args.table}; {args.games} games x '
          f'{args.positions_per_game} positions; {args.samples} samples; seed {args.seed}')
    positions = collect_positions(table, args.games, args.positions_per_game, args.seed)
    rows = []
    for index, position in enumerate(positions):
        seed = stream_seed(args.seed, 'accuracy-search', index)
        replicate = stream_seed(args.seed, 'accuracy-search-replicate', index) if args.rank_check else None
        rows.append(dict(game=position['game'], seat=position['obs'].seat, action_id=position['action_id'],
                         tricks_completed=len(position['obs'].view.completed_tricks),
                         realized_marks=position['realized_marks'],
                         beliefs=belief_scores(position, factories, args.samples, seed),
                         values=value_scores(position, factories, leaf, args.samples, seed, replicate)))
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
    if 'ranking' in summary:
        print('\nMove ranking: candidate vs reference search on the same worlds and seeds')
        print(f'{"comparison":11s} {"concord":>8s} {"top-1":>6s} {"spearman":>9s} {"regret":>8s}  units')
        for name, s in summary['ranking'].items():
            concordance = '-' if s['concordance'] is None else f'{s["concordance"]:.1%}'
            spearman = '-' if s['spearman'] is None else f'{s["spearman"]:.3f}'
            print(f'{name:11s} {concordance:>8s} {s["top1"]:6.1%} {spearman:>9s} {s["regret"]:8.3f}  '
                  f'{s["regret_units"]} ({s["candidate"]} vs {s["reference"]})')
    if args.output:
        data = dict(model=dict(path=identity.path, sha256=identity.sha256), table=args.table, games=args.games,
                    positions_per_game=args.positions_per_game, samples=args.samples, seed=args.seed,
                    smoothing=SMOOTHING, summary=summary, rows=rows)
        if partner is not None:
            data['partner_model'] = dict(path=partner.path, sha256=partner.sha256)
        if args.rank_check:
            data['rank_check'] = True
        args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'\nJSON results saved to {args.output}')
    # record_output flattens a returned list into the per-position CSV export.
    return rows


if __name__ == '__main__':
    main()
