"""Builds the QB Forecast Lab logo SVGs (public/brand, src/app/icon.svg). Wordmark outlined from Barlow Condensed SemiBold (SIL OFL 1.1).

Usage: python scripts/build-brand.py . <path to BarlowCondensed-SemiBold .ttf or .woff2>   (needs fontTools; brotli for .woff2)
PNG icons (src/app/apple-icon.png, public/brand/icon-192.png, icon-512.png) are rasterized from public/brand/app-icon.svg.
"""
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FE = Path(sys.argv[1]); FONT = sys.argv[2]
OUT = FE / "public" / "brand"; OUT.mkdir(parents=True, exist_ok=True)

# mark geometry, 58 x 40 units: a quarterback's throwing arc (release point + solid flight), the projected remainder
# as two dots, and the landing zone drawn as a forecast range: an 80% band with its 50% core, like the site's field ruler
def mark(ink, acc, band, dots=True):
    d = (f'<path d="M5 29Q19.4 9.8 33.8 13.64" fill="none" stroke="{ink}" stroke-width="3.8" stroke-linecap="round"/>')
    if dots:
        d += f'<circle cx="41" cy="17" r="2.05" fill="{acc}"/><circle cx="47.24" cy="22.24" r="2.05" fill="{acc}"/>'
    d += (f'<rect x="42" y="31" width="15" height="5.2" rx="1.4" fill="{band}"/>'
          f'<rect x="46" y="31" width="7.6" height="5.2" rx="1.4" fill="{acc}"/>'
          f'<circle cx="5" cy="29" r="3.4" fill="{ink}"/>')
    return d

DARK = dict(ink="#E9E5DB", acc="#4C8BD1", band="#2E527E")     # on field navy
LIGHT = dict(ink="#0F1A2B", acc="#2F6DB5", band="#A9C3E3")    # on white / paper
TEXT = {"dark": "#E9E5DB", "light": "#0F1A2B"}

font = TTFont(FONT); gs = font.getGlyphSet(); cmap = font.getBestCmap(); upm = font["head"].unitsPerEm
hmtx = font["hmtx"]

def wordmark(text, size, x0, baseline, tracking=0.0):
    s = size / upm; x = x0; parts = []
    for ch in text:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, baseline)))
        if pen.getCommands(): parts.append(pen.getCommands())
        x += hmtx[g][0] * s + tracking * size
    return " ".join(parts), x - tracking * size

def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}" width="{w:.0f}" height="{h:.0f}" '
            f'role="img" aria-label="{title}"><title>{title}</title>{body}</svg>\n')

# horizontal lockup: font size 30, mark scaled 0.8 so the band sits on the baseline and the arc tops at cap height
FS, SC, GAP = 30.0, 0.8, 10.0
cap = font["OS/2"].sCapHeight * FS / upm
base = 6 + cap  # 6 units of padding above cap height
mark_top = base - 36.2 * SC
for theme, pal in (("dark", DARK), ("light", LIGHT)):
    path, end = wordmark("QB Forecast Lab", FS, 4 + 58 * SC + GAP, base)
    W, H = end + 4, base + 8
    body = (f'<g transform="translate(4 {mark_top:.2f}) scale({SC})">{mark(**pal)}</g>'
            f'<path d="{path}" fill="{TEXT[theme]}"/>')
    (OUT / f"logo-horizontal-{theme}.svg").write_text(svg(W, H, body, "QB Forecast Lab"), encoding="utf-8")
    (OUT / f"mark-{theme}.svg").write_text(svg(58, 40, mark(**pal), "QB Forecast Lab"), encoding="utf-8")

# app icon: a heavier cut of the same geometry for 16-512 px: thicker arc, one projection dot, a taller range band
def icon_mark(ink, acc, band):
    return (f'<path d="M5.5 30Q21.32 7.74 37.6 15.4" fill="none" stroke="{ink}" stroke-width="5.4" stroke-linecap="round"/>'
            f'<circle cx="46.3" cy="21.6" r="3" fill="{acc}"/>'
            f'<rect x="39.5" y="29" width="18" height="8" rx="2" fill="{band}"/>'
            f'<rect x="44" y="29" width="9" height="8" rx="2" fill="{acc}"/>'
            f'<circle cx="5.5" cy="30" r="4.4" fill="{ink}"/>')

def tile(radius=14):
    sc = 50 / 58; tx = (64 - 58 * sc) / 2; ty = (64 - 40 * sc) / 2 + 2
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            f'<rect width="64" height="64" rx="{radius}" fill="#0F1A2B"/>'
            f'<g transform="translate({tx:.2f} {ty:.2f}) scale({sc:.4f})">{icon_mark(**DARK)}</g></svg>\n')
(OUT / "icon-mark-dark.svg").write_text(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 58 40">{icon_mark(**DARK)}</svg>\n', encoding="utf-8")
(FE / "src" / "app" / "icon.svg").write_text(tile(), encoding="utf-8")
(OUT / "app-icon.svg").write_text(tile(radius=0), encoding="utf-8")  # full bleed: platforms mask it
print("cap", cap, "files", sorted(p.name for p in OUT.iterdir()))
