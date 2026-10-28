"""Predict next-season QB YPA using only prior-season stats. Run after pipeline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['ypa', 'td_rate', 'int_rate', 'attempts', 'era_score']


def make_examples(df: pd.DataFrame) -> pd.DataFrame:
    data = df.sort_values(['player_id', 'season']).copy()
    data['target_ypa'] = data.groupby('player_id')['ypa'].shift(-1)
    data['target_season'] = data.groupby('player_id')['season'].shift(-1)
    # IMPORTANT: only consecutive seasons, never skip an absent year.
    data = data[data.target_season.eq(data.season + 1)].copy()
    # Require meaningful next-year passing volume to reduce noisy targets.
    data['target_attempts'] = df.sort_values(['player_id', 'season']).groupby('player_id')['attempts'].shift(-1).loc[data.index]
    return data[data.target_attempts >= 100].copy()


def report(name: str, model, x, y):
    pred = model.predict(x)
    return {'model': name, 'n': len(y), 'MAE': round(float(mean_absolute_error(y, pred)), 4),
            'RMSE': round(float(np.sqrt(mean_squared_error(y, pred))), 4)}


def main():
    src = ROOT / 'data' / 'qb_seasons.csv'
    if not src.exists():
        raise SystemExit('First run: python -m src.pipeline')
    data = make_examples(pd.read_csv(src))
    # Each row's predictors are from season t; target is season t+1.
    train = data[data.target_season <= 2021]
    valid = data[data.target_season.between(2022, 2023)]
    test = data[data.target_season.between(2024, 2025)]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError('Insufficient samples in temporal train/validation/test periods')
    scores = []
    for alpha in [0.1, 1, 10, 100]:
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha))
        model.fit(train[FEATURES], train.target_ypa)
        scores.append((mean_absolute_error(valid.target_ypa, model.predict(valid[FEATURES])), alpha))
    alpha = min(scores)[1]
    # Refitting after tuning on validation data is allowed; test stays untouched.
    trainval = pd.concat([train, valid], ignore_index=True)
    ridge = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha))
    ridge.fit(trainval[FEATURES], trainval.target_ypa)
    baseline = DummyRegressor(strategy='mean').fit(trainval[FEATURES], trainval.target_ypa)
    # Strong additional baseline: predict same YPA as preceding season.
    persistence = {'model': 'Prior-season YPA', 'n': len(test),
                   'MAE': round(float(mean_absolute_error(test.target_ypa, test.ypa)), 4),
                   'RMSE': round(float(np.sqrt(mean_squared_error(test.target_ypa, test.ypa))), 4)}
    metrics = {'chosen_alpha': alpha, 'splits': {'train':len(train), 'validation':len(valid), 'test':len(test)},
               'test_results':[report('Mean baseline', baseline, test[FEATURES], test.target_ypa),
                               persistence, report('Ridge model', ridge, test[FEATURES], test.target_ypa)]}
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    (dest / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
