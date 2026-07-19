"""
Stochastic ensemble sampler for the two-stage capacity-expansion phase.

Draws a flat, equal-weighted set of climate futures for the cost-of-inaction
study. Each sample keeps the full 82-year historical inflow sequence intact and
perturbs it with a single smooth multiplier surface (via
``futures.apply_warm_shift``); the only randomness is the pair

    (WA, WI) = (total rim-inflow change, Nov-Apr winter-fraction ratio)

drawn from a bivariate distribution fit to the California Fourth Assessment GCM
cloud (the 20 GCM x RCP members shipped under
``data/fourth-assessment-data``). The applied monthly shape is the
ensemble-mean surface rescaled to the drawn (WA, WI), so a sample is one point on
a smooth perturbed inflow surface, not a resampled sequence (no block bootstrap).

The Colorado River import cut rides the same dryness draw: ``colorado_cut_from_wa``
maps a drier WA to a larger cut, anchored to the study's likely/worse magnitudes
(~500 TAF/yr at median dryness, ~1000 TAF/yr at the WA=-8.5% tail).

Design: ``notes/01-design/cost-of-inaction-study-design.md`` (revised to a single
joint ensemble). The module is df-pure and deterministic given a seed; run-specific
knobs (sample count, seed, tail inflation, Colorado anchors) live in the analysis
runners, the way ``FUTURES``/``T_LADDER`` live in the sweep scripts.
"""
import os

import numpy as np
import pandas as pd

from calvin.futures import (apply_warm_shift, rim_inflow_mask, GCM_DIR,
                            WINTER_MONTHS)


# ---------------------------------------------------------------------------
# The GCM (WA, WI) cloud
# ---------------------------------------------------------------------------
def gcm_members():
  """List ``(member, rcp)`` for the 20 shipped per-GCM multiplier files."""
  out = []
  for fn in sorted(os.listdir(GCM_DIR)):
    if not fn.endswith('.csv') or fn == 'overall.csv':
      continue
    member, _, rcp = fn[:-len('.csv')].rpartition('.')
    out.append((member, rcp))
  return out


def wa_wi_cloud(base_links_df, *, cache_path=None):
  """
  The empirical (WA, WI) cloud of the 20 GCM x RCP members.

  For each member, apply its raw monthly multipliers to ``base_links_df`` with no
  aggregate rescale (``wa_target=None``) and read back the realized aggregate
  water-availability change and winter-fraction ratio. This lands the cloud in the
  exact (WA, WI) space that later draws are rescaled *to*, so the fit and the
  applied perturbation share one metric.

  :param base_links_df: a links DataFrame carrying the rim INFLOW arcs (the full
    82-yr record for the production cloud; a smoke slice is fine for tests).
  :param cache_path: optional CSV path; loaded if it exists, else written.
  :returns: DataFrame indexed by ``'<member>.<rcp>'`` with columns ``wa, wi``.
  """
  if cache_path and os.path.isfile(cache_path):
    return pd.read_csv(cache_path, index_col=0)

  rows = {}
  for member, rcp in gcm_members():
    df = base_links_df.copy()
    _, info = apply_warm_shift(df, member=member, rcp=rcp,
                               wa_target=None, winter_index=None)
    rows['%s.%s' % (member, rcp)] = {'wa': info['realized_wa'],
                                     'wi': info['realized_winter_index']}
  cloud = pd.DataFrame.from_dict(rows, orient='index')[['wa', 'wi']]
  cloud.index.name = 'member'
  if cache_path:
    cloud.to_csv(cache_path)
  return cloud


def fit_joint(cloud):
  """Fit a bivariate normal to the (WA, WI) cloud -> (mean(2), cov(2,2)).

  A normal (not a KDE): 20 points is too few for a stable 2-D density, and CVaR
  needs smooth extrapolation into the dry tail rather than interpolation of the
  cloud. The covariance's off-diagonal preserves the observed WA-WI correlation
  (the seasonal-shift-with-drying signal).
  """
  x = cloud[['wa', 'wi']].to_numpy(dtype=float)
  return x.mean(axis=0), np.cov(x, rowvar=False)


# ---------------------------------------------------------------------------
# Drawing samples
# ---------------------------------------------------------------------------
def draw_wa_wi(n, rng, mean, cov, *, tail_inflation=1.0, mean_shift=(0.0, 0.0),
               wa_bounds=(-0.30, 0.30), wi_floor=0.5):
  """Draw ``n`` (WA, WI) pairs from the fitted normal.

  ``tail_inflation`` scales the covariance to probe deeper tails than 20 GCMs
  sample; ``mean_shift`` recenters. WA is clipped so ``1 + WA > 0`` and WI to a
  floor; the per-draw upper WI feasibility ceiling is applied in
  :func:`draw_samples` (it depends on the record's winter fraction).
  """
  m = np.asarray(mean, dtype=float) + np.asarray(mean_shift, dtype=float)
  draws = rng.multivariate_normal(m, np.asarray(cov, dtype=float) * tail_inflation,
                                  size=n)
  wa = np.clip(draws[:, 0], wa_bounds[0], wa_bounds[1])
  wi = np.maximum(draws[:, 1], wi_floor)
  return np.column_stack([wa, wi])


def colorado_cut_from_wa(wa, *, wa_lo=-0.042, cut_lo=500.0,
                         wa_hi=-0.085, cut_hi=1000.0,
                         cut_min=0.0, cut_max=2000.0,
                         noise_sigma=0.0, rng=None):
  """Monotone map from dryness to the Colorado import cut (TAF/yr).

  Linear in dryness ``d = -wa`` through the report's own likely/worse anchors:
  ``(wa_lo=-0.042 -> cut_lo=500)`` and ``(wa_hi=-0.085 -> cut_hi=1000)`` (Table 5).
  Wetter than likely ramps toward zero (a wet future needs no import cut); drier
  than worse extrapolates up to ``cut_max``. Fixed anchors, not cloud-dependent,
  so the ramp is stable regardless of the drawn ensemble. ``cut_max`` must stay
  under the annual import (``apply_colorado_cut`` raises otherwise). Optional
  Gaussian noise decouples the cut from dryness for imperfect correlation.
  """
  d, d_lo, d_hi = -wa, -wa_lo, -wa_hi
  slope = (cut_hi - cut_lo) / (d_hi - d_lo)
  cut = cut_lo + slope * (d - d_lo)
  if noise_sigma and rng is not None:
    cut = cut + rng.normal(0.0, noise_sigma)
  return float(np.clip(cut, cut_min, cut_max))


def _hist_winter_fraction(base_links_df):
  """Nov-Apr share of historical rim inflow (for the WI feasibility ceiling)."""
  mask = rim_inflow_mask(base_links_df)
  ub = base_links_df.loc[mask, 'upper_bound'].to_numpy(dtype=float)
  month = (base_links_df.loc[mask, 'i'].str.split('.').str[1]
           .str.split('-').str[1].astype(int).to_numpy())
  return ub[np.isin(month, WINTER_MONTHS)].sum() / ub.sum()


def draw_samples(n, seed, *, base_links_df, cloud=None, cache_path=None,
                 tail_inflation=1.0, mean_shift=(0.0, 0.0), colorado_kw=None):
  """
  Draw ``n`` future dicts, each ``{'warm_shift': {'wa_target', 'winter_index'},
  'colorado_cut_taf'}``, ready to pass to ``futures.apply_futures``.

  Reproducibility contract: one ``np.random.default_rng(seed)``; the whole WA-WI
  block is drawn first, then any Colorado noise in sample order. With no
  ``member``/``rcp`` in the dicts, ``apply_warm_shift`` uses the ensemble-mean
  monthly shape rescaled to each draw.

  :param base_links_df: the base links (used for the cloud if not supplied and for
    the winter feasibility ceiling).
  :param cloud: precomputed (WA, WI) cloud; computed from ``base_links_df`` if None.
  :param tail_inflation/mean_shift: covariance scaling / recentring knobs.
  :param colorado_kw: overrides for :func:`colorado_cut_from_wa` (e.g.
    ``noise_sigma``); ``wa_med`` defaults to the cloud median.
  """
  rng = np.random.default_rng(seed)
  if cloud is None:
    cloud = wa_wi_cloud(base_links_df, cache_path=cache_path)
  mean, cov = fit_joint(cloud)
  wawi = draw_wa_wi(n, rng, mean, cov, tail_inflation=tail_inflation,
                    mean_shift=mean_shift)

  winter_frac = _hist_winter_fraction(base_links_df)
  ckw = dict(colorado_kw or {})

  samples = []
  for wa, wi in wawi:
    wa, wi = float(wa), float(wi)
    wi = min(wi, (1.0 + wa) / winter_frac * 0.999)   # feasibility ceiling
    cut = colorado_cut_from_wa(wa, rng=rng, **ckw)
    samples.append({'warm_shift': {'wa_target': wa, 'winter_index': wi},
                    'colorado_cut_taf': cut})
  return samples


def mean_sample(cloud, *, colorado_kw=None):
  """The expected-value future (mean WA/WI, Colorado at mean WA, no noise) for the
  VSS reference build."""
  mean, _ = fit_joint(cloud)
  wa, wi = float(mean[0]), float(mean[1])
  ckw = dict(colorado_kw or {})
  ckw.pop('noise_sigma', None)                       # EV future is deterministic
  return {'warm_shift': {'wa_target': wa, 'winter_index': wi},
          'colorado_cut_taf': colorado_cut_from_wa(wa, **ckw)}
