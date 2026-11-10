"""Read-only analysis and visualization of completed SC42 experiments (docs/visualization.md).

This module holds only the plain-data figure catalog, so `--list-plots` and
argument validation work before the optional `viz` dependencies are imported.
"""

FORMATS = ('html', 'png', 'svg', 'pdf')
X_AXES = ('hands', 'matches', 'updates', 'runtime')

# name -> (what it shows, data it needs)
FAMILIES = {
    'learning': ('Held-out performance and training reward versus experience',
                 'progress.jsonl with `evaluations`/`evaluation`; `mean_reward` or `mean_signed_reward`'),
    'ppo': ('PPO optimization diagnostics, overall and per phase',
            'progress.jsonl `metrics` (or flat legacy fields) and `phase_samples`'),
    'critic': ('Critic loss, explained variance, value calibration, oracle auxiliary head',
               'progress.jsonl `metrics.value_loss`, `value_diagnostics`, `oracle_validation`'),
    'bidding': ('Bid/pass behaviour, attempts versus contracts, success by category',
                'progress.jsonl `bidding` and `rollout_bids`'),
    'match-bidding': ('Score-dependent bidding and match-level training progress',
                      'progress.jsonl `bidding.*.by_score_state`, `by_match_score`, `match_win_rate`, '
                      '`mean_hands_per_match`, `matches`'),
    'opponents': ('Training opponent exposure and identity',
                  'progress.jsonl `training_by_opponent`, `train_opponents`, `opponent_assignments`'),
    'evaluation': ('Held-out arena comparisons with recorded intervals',
                   'evaluate_agent_vs_agent / evaluate_selfplay_ppo / evaluate_ppo result JSON'),
    'contracts': ('Contract frequency, success, origin, decision timing and roles',
                  'arena report `breakdown`/`breakdowns`, or per-hand records'),
    'round-robin': ('Agent-versus-agent performance matrix',
                    'two or more evaluate_agent_vs_agent reports with `composition`'),
    'partners': ('First-seat by second-seat partnership compatibility',
                 'evaluate_agent_vs_agent reports with `composition.partners`'),
    'belief': ('Hidden-information inference quality', 'search_accuracy.py JSON `summary.beliefs` and `rows`'),
    'search': ('Search value accuracy, move ranking, cost, and search inside training',
               'search_accuracy.py JSON `summary.values`/`summary.ranking`; '
               'progress.jsonl `metrics.search_*` and `rollout_performance.search_*`'),
    'resources': ('Throughput, time split, search worker time, memory',
                  'progress.jsonl timing fields and `rollout_performance`'),
}

# Presets select figure families; they do not claim that the data answers the question.
RESEARCH_PRESETS = {
    'rq1': ('learning', 'ppo', 'critic', 'evaluation', 'resources'),
    'rq2': ('belief', 'search', 'contracts', 'evaluation', 'resources'),
    'rq3': ('partners', 'round-robin', 'bidding', 'opponents', 'evaluation'),
}
RESEARCH_QUESTIONS = {
    'rq1': 'Belief modeling, privileged-information training and PPO sample efficiency',
    'rq2': 'Necessity and effectiveness of explicit hidden-information inference',
    'rq3': 'Partnership coordination, specialization and compatibility with unfamiliar partners',
}
