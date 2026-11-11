"""Score drafted QBs with a saved career-success model.

Run:  python -m src.modeling.predict                      # censored 2022-2026 classes, model C + B + reference
      python -m src.modeling.predict --model B_context_l2 --draft-years 2024 2026

The 2022-2026 classes do not have complete five-season windows yet, so these
probabilities CANNOT be evaluated today. They are written for monitoring only;
the actual outcome (where already known) is printed next to them only to show
which are already resolved (a QB who has reached 32 starts is a known success).
By default a QB is scored only when he is inside the model's training support:
  * a play-type model needs his final team-season play-type profile (no imputing
    a whole feature group), and
  * every training QB played his final college season in FBS, so FCS QBs are
    not scored (their box-score rates against FCS defenses are an extrapolation).
--include-outside-support scores them anyway, with the flags kept in the output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import train as T

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'modeling' / 'predictions_censored_classes.csv'
DEFAULT_MODELS = ['C_playtype_l2', 'B_context_l2', 'draft_reference_l2']


def load_model(name: str, model_dir: Path = T.MODEL_DIR) -> dict:
    art = joblib.load(model_dir / f'{name}.joblib')
    assert {'pipeline', 'features', 'model_version'} <= set(art), 'unexpected artifact format'
    return art


def score(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    cols = art['features']
    out = df[['qb_draft_prospect_id', 'draft_year', 'player_name', 'college']].copy()
    out['model'] = art['model_name']
    out['model_version'] = art['model_version']
    out['n_features_missing'] = df[cols].isna().sum(axis=1).to_numpy()
    out['p_success'] = art['pipeline'].predict_proba(df[cols])[:, 1]
    out['predicted_success_at_threshold'] = (out['p_success'] >= art['threshold']).astype(int)
    return out


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', help='model name (repeatable); default: C, B and draft reference')
    ap.add_argument('--draft-years', nargs=2, type=int, default=[2022, 2026])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--include-outside-support', action='store_true')
    a = ap.parse_args(argv)
    data = T.load_data()
    lo, hi = a.draft_years
    rows = data[data['draft_year'].between(lo, hi) & data['has_college_box_features']].copy()
    res, skipped = [], {}
    for name in a.model or DEFAULT_MODELS:
        art = load_model(name)
        needs_pt = any(c.startswith('pt_') for c in art['features'])
        outside = rows['final_fcs'].fillna(False).astype(bool) | (needs_pt & ~rows['has_playtype_features'])
        use = rows if a.include_outside_support else rows[~outside]
        skipped[name] = int(len(rows) - len(use))
        s = score(use, art)
        s['outside_training_support'] = outside[use.index].to_numpy()
        res.append(s)
    out = pd.concat(res, ignore_index=True)
    out = out.merge(data[['qb_draft_prospect_id', 'nfl_starts_first_5_seasons_to_date', 'career_success',
                          'career_success_status']], on='qb_draft_prospect_id', how='left')
    out = out.rename(columns={'career_success': 'career_success_known_so_far'})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f'classes {lo}-{hi}: {len(rows)} QBs with college stats; skipped outside training support {skipped} '
          f'-> {a.out} (not evaluable yet)')
    return out


if __name__ == '__main__':
    main()
