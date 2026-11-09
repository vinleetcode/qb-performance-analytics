"""Tier-B scheme evidence: the `off_scheme` field of Wikipedia team-season articles.

Why this source, and why only LOW confidence:
  * The infobox field states the offensive scheme for that specific team-season, so it is not
    a projection of a coach's philosophy onto a new team.
  * It is community-edited and almost never cited, and it was usually written after the
    fact, so it is never treated as contemporaneous evidence. Labels from it are stored with
    confidence='low', source_type='wikipedia_infobox' and the exact revision permalink, so
    a model can drop or down-weight them.

Access: the official MediaWiki Action API (https://www.mediawiki.org/wiki/API:Main_page),
batched 50 titles per request, with a descriptive User-Agent and no personal data. Content
is CC BY-SA 4.0. Only the infobox field text and the revision metadata are stored. Every
response is cached under data/wikipedia_cache/ (gitignored), so reruns make no requests.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'wikipedia_cache'
API = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'qb-performance-analytics/0.3 (academic sports-analytics research; batched MediaWiki API reads)'
# [ \t]* (never \s*): an EMPTY field must not swallow the newline and capture the next line
FIELD_RE = {k: re.compile(r'^\|[ \t]*' + k + r'[ \t]*=[ \t]*(.*)$', re.M) for k in ('off_scheme', 'off_coach', 'head_coach')}
# Wikipedia article names that differ from "<season> <CFBD school> <CFBD mascot> football team"
# (checked against the 12 titles that were missing on the first pass).
TITLE_ALIASES = {
    'Troy': ['Troy Trojans'],
    'Miami (OH)': ['Miami RedHawks'],
    'Houston Christian': ['Houston Baptist Huskies'],
    "Hawai'i": ['Hawaii Warriors', 'Hawaii Rainbow Warriors'],
}


def _cached_get(params: dict) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
    path = CACHE / f'{key}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    for attempt in range(6):
        r = requests.get(API, params=params, headers={'User-Agent': USER_AGENT}, timeout=60)
        if r.status_code == 200:
            data = r.json()
            path.write_text(json.dumps(data), encoding='utf-8')
            time.sleep(5.0)  # be polite: full-content queries are rate-limited
            return data
        wait = float(r.headers.get('Retry-After') or 0) or 30 * (attempt + 1)
        time.sleep(min(wait, 180))
    r.raise_for_status()


def strip_wikitext(s: str | None) -> str | None:
    """'[[Run and shoot offense|Run and shoot]] <ref>..</ref>' -> 'Run and shoot'."""
    if s is None:
        return None
    s = re.sub(r'<ref[^>]*/>', '', s)
    s = re.sub(r'<ref[^>]*>.*?</ref>', '', s, flags=re.S)
    s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
    s = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', s)
    s = re.sub(r'\{\{[^}]*\}\}', '', s)
    s = re.sub(r"'{2,}", '', s).replace('<br>', ' / ').replace('<br />', ' / ').replace('<br/>', ' / ')
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,;')
    return s or None


def fetch_infoboxes(titles: list[str]) -> pd.DataFrame:
    """Infobox fields for each requested title (redirects followed). One row per
    requested title; missing pages have exists=False."""
    rows = []
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = _cached_get({'action': 'query', 'prop': 'revisions', 'rvprop': 'ids|timestamp|content',
                            'rvslots': 'main', 'titles': '|'.join(batch), 'redirects': 1,
                            'format': 'json', 'formatversion': 2})
        q = data.get('query', {})
        resolve = {}
        for kind in ('normalized', 'redirects'):
            for m in q.get(kind, []):
                resolve[m['from']] = m['to']
        pages = {p['title']: p for p in q.get('pages', [])}
        for t in batch:
            final = t
            while final in resolve:
                final = resolve[final]
            p = pages.get(final)
            if not p or p.get('missing') or not p.get('revisions'):
                rows.append({'requested_title': t, 'exists': False})
                continue
            rev = p['revisions'][0]
            txt = rev['slots']['main']['content']
            rec = {'requested_title': t, 'exists': True, 'page_title': p['title'], 'revid': rev['revid'],
                   'rev_timestamp': rev['timestamp']}
            for k, rx in FIELD_RE.items():
                m = rx.search(txt)
                raw = m.group(1).strip() if m else None
                rec[f'{k}_raw'] = raw
                rec[k] = strip_wikitext(raw)
            rec['off_scheme_has_citation'] = bool(rec['off_scheme_raw'] and '<ref' in rec['off_scheme_raw'])
            rows.append(rec)
    return pd.DataFrame(rows)


def season_page_title(season: int, school: str, mascot: str) -> str:
    return f'{season} {school} {mascot} football team'


def ascii_school(s: str) -> str:
    import unicodedata
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch)).replace("'", '')


def fetch_team_season_infoboxes(team_seasons: pd.DataFrame, mascots: pd.DataFrame) -> pd.DataFrame:
    """team_seasons: columns team, season (CFBD names). mascots: school, mascot (CFBD /teams).
    Tries the default title, then TITLE_ALIASES; returns one row per team-season."""
    ts = team_seasons.merge(mascots, left_on='team', right_on='school', how='left')
    ts['title'] = [season_page_title(s, ascii_school(t), m or '') for s, t, m in zip(ts['season'], ts['team'], ts['mascot'])]
    ib = fetch_infoboxes(ts['title'].tolist())
    out = ts.merge(ib, left_on='title', right_on='requested_title', how='left')
    miss = out[~out['exists'].fillna(False).astype(bool)]
    alt_rows = []
    for _, r in miss.iterrows():
        for alias in TITLE_ALIASES.get(r['team'], []):
            alt_rows.append({'team': r['team'], 'season': r['season'], 'title': f"{r['season']} {alias} football team"})
    if alt_rows:
        alts = pd.DataFrame(alt_rows)
        alt_ib = alts.merge(fetch_infoboxes(alts['title'].tolist()), left_on='title', right_on='requested_title')
        alt_ib = alt_ib[alt_ib['exists']].drop_duplicates(['team', 'season'])
        keep = out[~out.set_index(['team', 'season']).index.isin(alt_ib.set_index(['team', 'season']).index)]
        out = pd.concat([keep, alt_ib], ignore_index=True)
    return out.drop(columns=['school', 'mascot', 'requested_title'], errors='ignore')


def permalink(revid: int) -> str:
    return f'https://en.wikipedia.org/w/index.php?oldid={revid}'
