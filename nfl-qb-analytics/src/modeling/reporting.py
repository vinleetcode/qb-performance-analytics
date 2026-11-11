"""Small report helpers (no optional dependencies such as tabulate)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(v, floatfmt: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}' if abs(int(v)) >= 10000 else str(int(v))
    if isinstance(v, (float, np.floating)):
        return format(float(v), floatfmt)
    return str(v).replace('|', '/')


def md_table(df: pd.DataFrame, floatfmt: str = '.3f', index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    num = [pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]) for c in d.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---:' if n else '---' for n in num) + '|']
    for row in d.itertuples(index=False):
        lines.append('| ' + ' | '.join(_fmt(v, floatfmt) for v in row) + ' |')
    return '\n'.join(lines)
