"""Write figures, tables, the manifest and the offline HTML research report."""
import html
import json
from pathlib import Path

import pandas as pd

from .normalize import Dataset, normalized_training
from .plots import Figure, Omission, Styles, render_interactive, render_static

MANIFEST_FILE = 'visualization_manifest.json'
STATIC_FORMATS = ('png', 'svg', 'pdf')
GLOSSARY = {
    'learning': (('Held-out win rate', 'Fraction of hands (hand evaluations) or complete matches (match evaluations) won against a fixed opponent; hand and match rates are never drawn together.'),
                 ('Mean signed marks per hand', 'Marks won minus marks lost by the agent partnership per hand; the primary performance measure.'),
                 ('Training reward', 'Mean rollout reward. Its unit depends on the reward objective, so objectives have separate figures.')),
    'ppo': (('Policy loss', 'Clipped surrogate objective; a fall does not imply stronger play.'), ('Entropy', 'Spread of the action distribution, in nats.'),
            ('Approximate KL', 'How far an update moved the policy.'), ('Clip fraction', 'Share of samples with an active ratio clip.'), ('Gradient norm', 'Global L2 norm of the actor gradient.')),
    'critic': (('Value loss', 'Mean squared error of the value prediction. Unavailable for critic-free PPO.'), ('Explained variance', 'One minus the residual over return variance; 1 is perfect.'),
               ('Return estimator', 'GAE or Monte Carlo; part of every series label.')),
    'bidding': (('Bid attempt', 'A non-pass call at a bid decision.'), ('Winning contract', 'The contract a partnership actually won; a subset of attempts.'),
                ('Forced call', 'A bid with no legal pass.'), ('Contract success rate', 'Conditional on the contracts chosen, so not itself a measure of bidding quality.')),
    'match-bidding': (('Score state', 'Whether the partnership leads, is tied or trails in match marks. Hand-objective runs have no match score.'), ('Mean hands per match', 'Rollout hands divided by rollout matches, pooled.')),
    'opponents': (('Exposure', 'Training hands or matches against each named opponent.'), ('Identity', 'Recorded checkpoint hash; a name such as self-current is not one immutable policy.')),
    'evaluation': (('95% interval', 'Bootstrap interval recorded by the arena, resampling complete duplicate groups.'), ('Sample unit', 'Duplicate groups are the independent units; hands inside a match are not.')),
    'contracts': (('Decision timing', 'Trick at which a contract outcome became decided; only recorded positions count.'), ('n/a', 'No contracts in that category: undefined, not zero.')),
    'round-robin': (('Cell', 'Row agent versus column agent, from the row agent\'s perspective.'), ('Duplicate weights', 'Agents with the same checkpoint hash are merged, not treated as independent evidence.')),
    'partners': (('Original partnership', 'Partners established by recorded checkpoints as trained together.'), ('Cross partnership', 'Networks from different training runs; only recorded metadata classifies a pair.')),
    'belief': (('True-owner probability', 'Mean probability placed on the actual owner of each unseen domino.'), ('Log-loss', 'Smoothed negative log probability of the true owner.'), ('Effective sample size', 'Particle count that the belief weights amount to.')),
    'search': (('Value error', 'Accuracy against one realized outcome per position; includes irreducible noise.'), ('Ranking regret', 'Reference-measured cost of the candidate\'s top move; approximate.')),
    'resources': (('Worker seconds', 'Time summed across processes.'), ('Wall seconds', 'Elapsed time.')),
}


def _remove_previous(output_dir: Path) -> None:
    """Delete files a previous run listed, so stale figures of other selections cannot linger."""
    manifest = output_dir / MANIFEST_FILE
    if not manifest.is_file():
        return
    try:
        previous = json.loads(manifest.read_text(encoding='utf-8')).get('files', [])
    except ValueError:
        return
    root = output_dir.resolve()
    for relative in previous:
        path = (root / relative).resolve()
        if path.is_file() and path.is_relative_to(root):
            path.unlink()


def _cell(value):
    return json.dumps(value, separators=(',', ':')) if isinstance(value, (list, dict)) else value


def write_tables(data: Dataset, comparison: pd.DataFrame, directory: Path) -> list[str]:
    written = []
    directory.mkdir(parents=True, exist_ok=True)
    if data.training:
        pd.concat([normalized_training(data, name) for name in data.training], ignore_index=True).to_csv(directory / 'normalized_training.csv', index=False)
        written.append('normalized_training.csv')
    if not data.evaluation.empty:
        run_fields = dict(training_seed='seed', reward_objective='reward_objective', critic_mode='critic_mode', return_estimator='return_estimator', trained_phases='trained_phases')
        columns = {column: data.evaluation['experiment'].map(lambda name: _cell(data.meta(name).get(key))) for column, key in run_fields.items()}
        data.evaluation.assign(**columns).to_csv(directory / 'normalized_evaluation.csv', index=False)
        written.append('normalized_evaluation.csv')
    if len(data.experiments):
        comparison.to_csv(directory / 'experiment_comparison.csv', index=False)
        written.append('experiment_comparison.csv')
    return written


def _experiment_rows(data: Dataset) -> list[dict]:
    rows = []
    for name, meta in data.experiments.items():
        checkpoints = meta.get('checkpoints') or ({'checkpoint': meta['checkpoint_id']} if meta.get('checkpoint_id') else {})
        rows.append(dict(name=name, kind=meta['kind'], path=meta['path'], objective=meta.get('reward_objective'), critic=meta.get('critic_mode'),
                         estimator=meta.get('return_estimator'), seed=meta.get('seed'), hands=meta.get('final_hands'), matches=meta.get('final_matches'),
                         checkpoints={key: value[:12] for key, value in checkpoints.items()}))
    return rows


def write_report(path: Path, name: str, data: Dataset, figures: list[Figure], omissions: list[Omission], files: dict[str, list[str]], warnings: list[str], settings: dict) -> None:
    esc = lambda value: html.escape('' if value is None else str(value))
    families = list(dict.fromkeys([f.family for f in figures] + [o.family for o in omissions]))
    navigation = ''.join(f'<a href="#{esc(family)}">{esc(family)}</a>' for family in families)
    rows = ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in (r['name'], r['kind'], r['objective'], r['critic'], r['estimator'], r['seed'],
                                                                   None if r['hands'] is None else f'{r["hands"]:,.0f}', None if r['matches'] is None else f'{r["matches"]:,.0f}',
                                                                   ', '.join(f'{k}: {v}' for k, v in r['checkpoints'].items()), r['path'])) + '</tr>' for r in _experiment_rows(data))
    warning_items = ''.join(f'<li>{esc(w)}</li>' for w in warnings)
    sections = []
    for family in families:
        body = [f'<section id="{esc(family)}"><h2>{esc(family)}</h2>']
        terms = GLOSSARY.get(family)
        if terms:
            body.append('<dl>' + ''.join(f'<dt>{esc(t)}</dt><dd>{esc(d)}</dd>' for t, d in terms) + '</dl>')
        for figure in figures:
            if figure.family != family:
                continue
            outputs = files[figure.id]
            html_file = next((f for f in outputs if f.endswith('.html')), None)
            image = next((f for f in outputs if f.endswith(('.png', '.svg'))), None)
            view = (f'<iframe src="{esc(html_file)}" title="{esc(figure.title)}" loading="lazy"></iframe>' if html_file else
                    f'<img src="{esc(image)}" alt="{esc(figure.title)}">' if image else '')
            links = ' '.join(f'<a href="{esc(f)}">{esc(Path(f).suffix[1:])}</a>' for f in outputs)
            missing = ''.join(f'<li><b>{esc(k)}</b>: {esc(v)}</li>' for k, v in figure.unavailable.items())
            body.append(f'<article><h3>{esc(figure.title)}</h3>{view}<p>{esc(figure.caption)}</p><p class="links">{links}</p>'
                        + (f'<details><summary>Series left out</summary><ul>{missing}</ul></details>' if missing else '') + '</article>')
        family_omissions = [o for o in omissions if o.family == family]
        if family_omissions:
            body.append('<h3>Omitted figures</h3><ul class="omitted">' + ''.join(f'<li><code>{esc(o.id)}</code>: {esc(o.reason)} <i>Needs: {esc(o.required)}</i></li>' for o in family_omissions) + '</ul>')
        body.append('</section>')
        sections.append(''.join(body))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(name)}</title><style>
:root{{color-scheme:light;--ink:#1a1a19;--muted:#5d5c58;--line:#dedcd6;--surface:#fcfcfb;--accent:#2a78d6}}
body{{margin:0;background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main,header{{max-width:1120px;margin:0 auto;padding:16px}}h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:21px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}}h3{{font-size:16px;margin:20px 0 6px}}
nav{{position:sticky;top:0;background:var(--surface);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;flex-wrap:wrap;gap:12px;z-index:1}}nav a,.links a{{color:var(--accent)}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid var(--line);padding:4px 8px;text-align:left}}iframe{{width:100%;height:560px;border:1px solid var(--line);background:#fff}}img{{max-width:100%;height:auto;border:1px solid var(--line);background:#fff}}
dl{{display:grid;grid-template-columns:minmax(120px,220px) 1fr;gap:2px 16px;font-size:14px}}dt{{font-weight:600}}dd{{margin:0;color:var(--muted)}}.warn{{border-left:4px solid #eda100;padding:4px 12px;background:#fdf6e6}}.note{{color:var(--muted)}}
</style></head><body><header><h1>{esc(name)}</h1><p class="note">Generated read-only from recorded SC42 experiment data. Differences between runs are observational unless the design isolates a cause; see the warnings below.</p></header>
<nav><a href="#experiments">experiments</a>{navigation}</nav><main><section id="experiments"><h2>Experiments</h2><div class="scroll"><table><tr><th>Name</th><th>Kind</th><th>Objective</th><th>Critic</th><th>Return estimator</th><th>Seed</th><th>Hands</th><th>Matches</th><th>Checkpoint identity</th><th>Source</th></tr>{rows}</table></div>
{f'<div class="warn"><h3>Comparison warnings and data notes</h3><ul>{warning_items}</ul></div>' if warning_items else ''}
<p class="note">Selected plots: {esc(', '.join(settings['plots']))}. Formats: {esc(', '.join(settings['formats']))}. Primary x-axis: {esc(settings['x_axis'])}.</p></section>{''.join(sections)}</main></body></html>'''
    path.write_text(page, encoding='utf-8')


def write_outputs(data: Dataset, figures: list[Figure], omissions: list[Omission], comparison: pd.DataFrame, warnings: list[str], output_dir: Path, settings: dict, *, name: str, report: bool) -> dict:
    """Create the output directory: figures in each selected format, tables, report and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_previous(output_dir)
    formats = settings['formats']
    styles = Styles(data.experiments)
    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(exist_ok=True)
    produced: dict[str, list[str]] = {}
    for figure in figures:
        static = {extension: figure_dir / f'{figure.id}.{extension}' for extension in formats if extension in STATIC_FORMATS}
        if static:
            render_static(figure, styles, static)
        if 'html' in formats:
            render_interactive(figure, styles, figure_dir / f'{figure.id}.html')
        produced[figure.id] = [f'figures/{figure.id}.{extension}' for extension in formats]
    tables = write_tables(data, comparison, output_dir / 'tables')
    for figure in figures:
        if figure.table is not None:
            figure.table.to_csv(output_dir / 'tables' / f'{figure.id}.csv', index=False)
            tables.append(f'{figure.id}.csv')
    files = [file for outputs in produced.values() for file in outputs] + [f'tables/{table}' for table in tables]
    if 'html' in formats and figures:
        files.append('figures/plotly.min.js')
    if report:
        write_report(output_dir / 'index.html', name, data, figures, omissions, produced, warnings, settings)
        files.append('index.html')
    manifest = dict(schema_version=1, name=name, settings=settings, experiments=[{key: value for key, value in meta.items() if key != 'config'} for meta in data.experiments.values()],
                    figures=[dict(id=f.id, family=f.family, title=f.title, caption=f.caption, unavailable=f.unavailable, files=produced[f.id]) for f in figures],
                    omissions=[dict(id=o.id, family=o.family, reason=o.reason, required=o.required) for o in omissions],
                    warnings=warnings, files=files)
    (output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    return manifest
