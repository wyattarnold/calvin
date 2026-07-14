import os
import datetime

import json
import csv
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_dict_as_csv(data, filename):
  """
  Given nested dict `data`, write to CSV file
  where rows are timesteps and columns are links/nodes as appropriate.

  :param data: (dict) Nested dictionary of results data
  :param filename: (string) Output CSV filename
  :returns: nothing, but writes the output CSV file.
  """
  node_keys = sorted(data.keys())
  time_keys = sorted(data[node_keys[0]].keys())  # add key=int for integer timesteps

  with open(filename, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['date'] + node_keys)
    for t in time_keys:
      row = [t] + [data[k].get(t) or 0.0 for k in node_keys]
      writer.writerow(row)


def dict_insert(D, k1, k2, v, collision_rule = None):
  """
  Custom insertion into nested dictionary.
  Assign D[k1][k2] = v if those keys do not exist yet.
  If the keys do exist, follow instructions for collision_rule
  """
  if k1 not in D:
    D[k1] = {k2: v}
  elif k2 not in D[k1]:
    D[k1][k2] = v
  else:
    if collision_rule == 'sum':
      D[k1][k2] += v
    # elif collision_rule == 'max':
    #   if v is not None and (D[k1][k2] is None or v > D[k1][k2]):
    #     D[k1][k2] = v
    elif collision_rule == 'first':
      pass # do nothing, we already have the first value
    elif collision_rule == 'last':
      D[k1][k2] = v # replace
    else:
      raise ValueError('Keys [%s][%s] already exist in dictionary' % (k1,k2))


def _collect_links(links, model, year, F, S, E, SV, SC, PV, PC, OC,
                   D_up, D_lo, EOP_storage, demand_set, pwp_set, op_set, eop=None):
  """Accumulate per-link results from a solved model into result dicts.

  All dict arguments are modified in-place.  If *eop* is provided it is used
  to accumulate end-of-period storage totals (keyed by reservoir name).

  Shared by :func:`postprocess` and :class:`cosvfea._PostprocessCollector`.
  """
  for link in links:
    s = tuple(link[0:3])
    ub = float(link[6])
    unit_cost = float(link[3])
    v  = model.X[s].value if s in model.X else 0.0
    d1 = model.dual.get(model.limit_lower[s], 0.0) if s in model.limit_lower else 0.0
    d2 = model.dual.get(model.limit_upper[s], 0.0) if s in model.limit_upper else 0.0

    if '.' in link[0] and '.' in link[1]:
      n1, t1 = link[0].split('.')
      n2, t2 = link[1].split('.')
      is_storage_node = (n1 == n2)
      if is_storage_node:
        amplitude = float(link[4])
        is_EOP = False
    elif '.' in link[0] and link[1] == 'FINAL':
      n1, t1 = link[0].split('.')
      is_storage_node = True
      amplitude = 1
      is_EOP = True
    elif link[0] == 'DBUGSRC' and '.' in link[1]:
      n2, t2 = link[1].split('.')
      if year is not None and year > 1922:
        t2 = t2.replace('1922', str(year)).replace('1921', str(year - 1))
      dict_insert(F, 'DBUGSRC-' + n2, t2, v, 'sum')
      continue
    elif '.' in link[0] and link[1] == 'DBUGSNK':
      n1, t1 = link[0].split('.')
      if year is not None and year > 1922:
        t1 = t1.replace('1922', str(year)).replace('1921', str(year - 1))
      dict_insert(F, n1 + '-DBUGSNK', t1, v, 'sum')
      continue
    else:
      continue

    if year is not None and year > 1922:
      t1 = t1.replace('1922', str(year)).replace('1921', str(year - 1))

    if is_storage_node:
      key = n1
      if is_EOP:
        if eop is not None:
          eop[key] = eop.get(key, 0.0) + v
        dict_insert(EOP_storage, key, t1, v, 'sum')
        dict_insert(S, key, t1, v, 'sum')
      else:
        evap = (1 - amplitude) * float(v) / amplitude
        dict_insert(S, key, t1, v, 'sum')
        dict_insert(E, key, t1, evap, 'sum')
    else:
      key = n1 + '-' + n2
      dict_insert(F, key, t1, v, 'sum')

      if key in demand_set:
        if (ub - v) > 1e-6:
          dict_insert(SV, key, t1, ub - v, 'sum')
          dict_insert(SC, key, t1, -1 * unit_cost * (ub - v), 'sum')
        else:
          dict_insert(SV, key, t1, 0.0, 'sum')
          dict_insert(SC, key, t1, 0.0, 'sum')

      if key in pwp_set:
        if (ub - v) > 1e-6 and unit_cost < 0:
          dict_insert(PV, key, t1, ub - v, 'sum')
          dict_insert(PC, key, t1, -1 * unit_cost * (ub - v), 'sum')
        else:
          dict_insert(PV, key, t1, 0.0, 'sum')
          dict_insert(PC, key, t1, 0.0, 'sum')

      if key in op_set:
        dict_insert(OC, key, t1, unit_cost * v, 'sum')

    dict_insert(D_up, key, t1, d1, 'last')
    dict_insert(D_lo, key, t1, d2, 'first')


def postprocess(df, model, resultdir=None, annual=False, year=None):
  """
  Postprocess model results into timeseries CSV files.

  :param df: (dataframe) network links data
  :param model: (CALVIN object) model object, post-optimization
  :param resultdir: (string) directory to place CSV file results
  :param annual: (boolean) whether to run annual optimization or not
  :param year: (int) for annual cosvf runs, the water year being postprocessed
  :returns: (dict) end-of-period storage keyed by node name (suitable for
    passing as ``ic`` to the next year's CALVIN run), only when annual=True.
    Otherwise writes result CSVs and returns None.
  """
  F, S, E, SV, SC, PV, PC, OC = {}, {}, {}, {}, {}, {}, {}, {}
  D_up, D_lo, D_node = {}, {}, {}
  EOP, EOP_storage = {}, {}

  links = df.values
  nodes = pd.unique(df[['i', 'j']].values.ravel()).tolist()
  demand_nodes = pd.read_csv(os.path.join(BASE_DIR, "data", "demand_nodes.csv"), index_col=0)
  pwp_nodes    = pd.read_csv(os.path.join(BASE_DIR, "data", "pwp_nodes.csv"),    index_col=0)
  op_nodes     = pd.read_csv(os.path.join(BASE_DIR, "data", "operation_nodes.csv"), index_col=0)

  _collect_links(
    links, model, year,
    F, S, E, SV, SC, PV, PC, OC, D_up, D_lo, EOP_storage,
    set(demand_nodes.index), set(pwp_nodes.index), set(op_nodes.index),
    eop=EOP,
  )

  # Remove any DBUGSRC inflows from GW end-of-period storage so that
  # debug-added water does not propagate to the next year as an inflated IC.
  for key in list(EOP.keys()):
    if key.startswith('GW_'):
      dbug_key = 'DBUGSRC-' + key
      total_dbug = sum(F.get(dbug_key, {}).values())
      if total_dbug > 0:
        EOP[key] = max(0.0, EOP[key] - total_dbug)
        for t in EOP_storage.get(key, {}):
          EOP_storage[key][t] = max(0.0, EOP_storage[key][t] - total_dbug)

  # get dual values for nodes (mass balance)
  for node in nodes:
    if '.' in node:
      n3, t3 = node.split('.')
      d3 = model.dual.get(model.flow[node], 0.0) if node in model.flow else 0.0
      dict_insert(D_node, n3, t3, d3)

  # write the output files
  if not resultdir:
    if annual:
      raise RuntimeError('resultdir must be specified for annual run')
    resultdir = 'results-' + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
  if not os.path.isdir(resultdir):
    os.makedirs(resultdir)

  things_to_save = [(F, 'flow'), (S, 'storage'), (D_up, 'dual_upper'),
                    (D_lo, 'dual_lower'), (D_node, 'dual_node'),
                    (E, 'evaporation'), (SV, 'shortage_volume'),
                    (SC, 'shortage_cost'), (EOP_storage, 'eop_storage'),
                    (PV, 'pwp_short_volume'), (PC, 'pwp_short_cost'),
                    (OC, 'operation_costs')]

  for data, name in things_to_save:
    save_dict_as_csv(data, resultdir + '/' + name + '.csv')

  if not annual:
    aggregate_regions(resultdir)
  else:
    return EOP


def combine_annual_results(years, annual_dir, output_dir=None):
  """
  Combine per-year result directories from a limited-foresight annual run
  into single concatenated CSV files.

  :param years: iterable of water years (ints), e.g. range(1922, 2004)
  :param annual_dir: directory containing per-year subdirectories named
    ``WY{year}`` (e.g. ``results/annual``)
  :param output_dir: directory to write combined CSVs. Defaults to the
    parent of ``annual_dir``.
  :returns: output_dir path (string)
  """
  years = list(years)
  annual_dir = os.path.abspath(annual_dir)
  if output_dir is None:
    output_dir = os.path.dirname(annual_dir)
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  # Discover CSV names from the first available year directory
  csv_names = set()
  for year in years:
    year_dir = os.path.join(annual_dir, 'WY%d' % year)
    if os.path.isdir(year_dir):
      csv_names = {os.path.splitext(f)[0] for f in os.listdir(year_dir) if f.endswith('.csv')}
      break

  for name in sorted(csv_names):
    frames = []
    for year in years:
      fp = os.path.join(annual_dir, 'WY%d' % year, name + '.csv')
      if os.path.exists(fp):
        frames.append(pd.read_csv(fp, index_col=0, parse_dates=True))
    if frames:
      combined = pd.concat(frames).sort_index()
      combined.to_csv(os.path.join(output_dir, name + '.csv'))

  return output_dir


def aggregate_regions(fp):
  """
  Read the results CSV files and aggregate results by region (optional).

  :param fp: (string) directory where output files are written.
  :returns: nothing, but overwrites the results files with new
    columns added for regional aggregations.
  """

  # aggregate regions and supply portfolios
  # easier to do this with pandas by just reading the CSVs again
  sc = pd.read_csv(fp + '/shortage_cost.csv', index_col=0, parse_dates=True)
  sv = pd.read_csv(fp + '/shortage_volume.csv', index_col=0, parse_dates=True)
  flow = pd.read_csv(fp + '/flow.csv', index_col=0, parse_dates=True)
  demand_nodes = pd.read_csv(os.path.join(BASE_DIR, "data", "demand_nodes.csv"), index_col = 0)
  portfolio = pd.read_csv(os.path.join(BASE_DIR, "data", "portfolio.csv"), index_col = 0)

  def _add_aggregated_columns(df, lookup, *group_keys):
    """For each combination of group_keys in lookup, sum the matching df columns."""
    for key_vals, group_rows in lookup.groupby(list(group_keys)):
      vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
      col_name = '_'.join(str(v) for v in vals)
      df[col_name] = df[group_rows.index].sum(axis=1)

  _add_aggregated_columns(sc,   demand_nodes, 'region', 'type')
  _add_aggregated_columns(sv,   demand_nodes, 'region', 'type')
  _add_aggregated_columns(flow, portfolio,    'region', 'supplytype', 'type')

  sc.to_csv(fp + '/shortage_cost.csv')
  sv.to_csv(fp + '/shortage_volume.csv')
  flow.to_csv(fp + '/flow.csv')
