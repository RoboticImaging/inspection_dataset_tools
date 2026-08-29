# Benchmark analysis

Everything here is generated from `collacted_results/{cloud,depth,pose}.csv` (4,048 rows per metric
category, written by `collate_results.py`). Nothing is hand-entered.

```bash
venv/bin/python analysis/fig_coverage.py    # figures 01-02
venv/bin/python analysis/fig_methods.py     # figures 03-06
venv/bin/python analysis/fig_axes.py        # figures 07-14, A1-A4
venv/bin/python analysis/fig_assets.py      # figures 15-21
venv/bin/python analysis/fig_frames.py      # figure  22
venv/bin/python analysis/tables.py          # tables/tables.md + tables/tables.tex
venv/bin/python analysis/webimg.py          # downscaled jpgs for the html report
venv/bin/python analysis/build_report.py    # the html report
```

`figures/` holds every plate as **both PNG (200 dpi) and PDF** — use the PDF in LaTeX.

| file | what it is |
|---|---|
| `common.py` | loading, the `ok` / `OOM` / `missing` status column, derived `Standoff` and `Family` |
| `style.py` | palette, matplotlib defaults, shared title/legend/heatmap helpers |
| `validate_palette.py` | Python port of the data-viz colour checks (no node in this env); reproduces the reference palette's documented numbers exactly |

## Conventions the figures follow

- **Status.** `OOM` = the run hit GPU out-of-memory; blank = never run. Both become `NaN` on load and
  are drawn as `OOM` / `--`, never as a zero.
- **Default scope** is `CORE_ASSETS` = glass_house2, old_house, unfinished_build — the three assets
  all eleven methods completed. Figures that widen to all seven say so in the subtitle, and the
  `A*` figures are the widened versions of the axis plates.
- **Aggregation** is median within an asset, then mean across assets, so glass_house2's 144 variants
  cannot outweigh bridge's 16.
- **Trajectory** is re-expressed as `near / mid / far / trendspek`: each asset's three
  `proxy_surface_<n>` offsets differ in absolute metres (bridge 5/10/15, crane 0/2/4, ...) so the raw
  name is not comparable across assets; the within-asset rank is.
- **Soil and reflectivity are separate render blocks**, not one grid. `soil=verylow` and
  `reflectivity=0.0` are different renders and do not agree numerically — never pool them.

## Findings, in one line each

1. OOM is a clean function of sequence length; every one-shot method has a ceiling between 100 and
   800 frames, no chunked/streaming method has one.
2. `da3_streaming` gives up almost nothing to `da3` and scales to 6/7 assets instead of 3.
3. Rotation accuracy is saturated; only RTA@5° and AUC@30 discriminate.
4. Cloud precision at 5 cm is floored by GT sampling density for every method — report F@0.25 m.
5. The large outdoor assets are ~10× harder on scale-free metrics; `crane` is the hardest.
6. The real `trendspek` trajectory is far harder than the synthetic orbits, and the gap grows with
   sequence length.
7. Soiling moves nothing for 9/11 methods; `mpsfm` alone collapses (AUC 0.815 → 0.343).
8. Reflectivity is the same story but also hits the VGGT family — the split is by how a method
   establishes correspondence, not learned vs classical.
9. Illumination sensitivity is per-method, and *sunrise* is the worst condition for 8/11 — not night.
10. **glass_house2 / sunrise / near+mid stand-off collapses all ten learned methods while COLMAP is
    unaffected.** Check those forty renders before writing this up either way.
11. **COLMAP and VGGT-SLAM are scored only on the frames they registered** (COLMAP mean 83%, worst
    31%, and it drops most where the scene is hardest). Every method comparison needs this caveat.
12. office2 is effectively absent from the results; draw nothing from it.
