"""Python port of the dataviz skill's validate_palette.js (no node in this env).

Same thresholds, same Machado-Oliveira-Fernandes 2009 severity-1.0 CVD matrices,
same OKLab Delta E x100 metric -- so the numbers are comparable to the skill's.
"""
import math, sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def _s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    h = h.strip().lstrip("#")
    return [_s2lin(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent", verbose=True):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True
    rep = []

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    rep.append(("Lightness band", not off, f"outside {lo}-{hi}: {off}" if off else f"all {len(palette)} inside"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    rep.append(("Chroma floor", not lowc, f"below floor: {lowc}" if lowc else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    pl = ([(i, j) for i in range(n) for j in range(i + 1, n)] if pairs == "all"
          else [(i, i + 1) for i in range(n - 1)])

    worst_cvd, worst_cvd_pair, warns = 1e9, None, []
    for i, j in pl:
        d = min(delta_e(palette[i], palette[j], "protan"), delta_e(palette[i], palette[j], "deutan"))
        if d < worst_cvd:
            worst_cvd, worst_cvd_pair = d, (palette[i], palette[j])
        if CVD_FLOOR <= d < CVD_TARGET:
            warns.append((palette[i], palette[j], round(d, 1)))
    cvd_ok = worst_cvd >= CVD_FLOOR
    ok &= cvd_ok
    rep.append((f"CVD separation ({pairs})", cvd_ok,
                f"worst dE {worst_cvd:.1f} {worst_cvd_pair}" + (f"; WARN band {warns}" if warns else "")))

    worst_n, worst_n_pair = 1e9, None
    for i, j in pl:
        d = delta_e(palette[i], palette[j])
        if d < worst_n:
            worst_n, worst_n_pair = d, (palette[i], palette[j])
    n_ok = worst_n >= NORMAL_FLOOR
    ok &= n_ok
    rep.append(("Normal-vision floor", n_ok, f"worst dE {worst_n:.1f} {worst_n_pair} (>= {NORMAL_FLOOR})"))

    low_contrast = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    rep.append(("Contrast vs surface", not low_contrast,
                f"WARN relief rule (needs labels/table): {low_contrast}" if low_contrast else f"all >= {CONTRAST_MIN}:1"))

    if verbose:
        print(f"--- mode={mode} surface={surface} pairs={pairs} n={n}")
        for name, passed, msg in rep:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:26s} {msg}")
        print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    mode = "light"
    pairs = "adjacent"
    surface = None
    args = sys.argv[2:]
    for k, v in zip(args, args[1:] + [""]):
        if k == "--mode": mode = v
        if k == "--pairs": pairs = v
        if k == "--surface": surface = v
    sys.exit(0 if validate(pal, mode=mode, surface=surface, pairs=pairs) else 1)
