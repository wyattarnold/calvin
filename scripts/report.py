"""
Generate a standard markdown results report for a CALVIN model run.

Can be run standalone (set MODEL_DIR below) or imported and called from a
run script:

  from report import generate_report
  generate_report('./my-models/calvin-pf')

Output:
  {MODEL_DIR}/report.md
  {MODEL_DIR}/figures/shortage_timeseries.png
  {MODEL_DIR}/figures/storage_timeseries.png
  {MODEL_DIR}/figures/supply_portfolio.png
"""
import os
import datetime
import pandas as pd
from calvin.plots import shortage_timeseries, storage_timeseries, supply_portfolio
from calvin.network.prepare import DEFAULT_R_TYPE1

# Default model directory when run as a standalone script
MODEL_DIR = './my-models/calvin-pf'

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR  = os.path.join(_REPO_ROOT, 'calvin', 'data')


def generate_report(model_dir):
    """Generate a markdown report and figures for a completed CALVIN run.

    :param model_dir: path to the model directory (must contain a results/ subdir).
    """
    result_dir  = os.path.join(model_dir, 'results')
    fig_dir     = os.path.join(model_dir, 'figures')
    report_path = os.path.join(model_dir, 'report.md')
    model_name  = os.path.basename(os.path.abspath(model_dir))

    os.makedirs(fig_dir, exist_ok=True)

    # Load results
    print('Loading results...')
    sc      = pd.read_csv(os.path.join(result_dir, 'shortage_cost.csv'),    index_col=0, parse_dates=True)
    sv      = pd.read_csv(os.path.join(result_dir, 'shortage_volume.csv'),  index_col=0, parse_dates=True)
    storage = pd.read_csv(os.path.join(result_dir, 'storage.csv'),          index_col=0, parse_dates=True)
    flow    = pd.read_csv(os.path.join(result_dir, 'flow.csv'),             index_col=0, parse_dates=True)
    oc      = pd.read_csv(os.path.join(result_dir, 'operation_costs.csv'),  index_col=0, parse_dates=True)

    portfolio = pd.read_csv(os.path.join(_DATA_DIR, 'portfolio.csv'), index_col=0)
    op_groups = pd.read_csv(os.path.join(_DATA_DIR, 'operation_groups.csv'), index_col=0)

    # Helpers
    def _link_cols(df):
        """Exclude regional aggregate columns (_urban/_ag) to avoid double-counting."""
        return [c for c in df.columns if not (c.endswith('_urban') or c.endswith('_ag'))]

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    sc_annual = sc[_link_cols(sc)].sum(axis=1).resample('YE-SEP').sum() / 1e3  # $M/yr
    sv_annual = sv[_link_cols(sv)].sum(axis=1).resample('YE-SEP').sum()        # TAF/yr
    oc_annual = oc.sum(axis=1).resample('YE-SEP').sum() / 1e3                  # $M/yr

    n_years      = len(sc_annual)
    period_start = sc.index[0].strftime('%b %Y')
    period_end   = sc.index[-1].strftime('%b %Y')
    mean_sc      = sc_annual.mean()
    mean_sv      = sv_annual.mean()
    mean_oc      = oc_annual.mean()

    gw_cols  = [c for c in storage.columns if c.startswith('GW_')]
    sr_cols  = [c for c in storage.columns if c in DEFAULT_R_TYPE1]
    gw_eop   = storage[gw_cols].resample('YE-SEP').last()
    gw_ic    = gw_eop.iloc[0]
    gw_final = gw_eop.iloc[-1]
    gw_od    = ((gw_ic - gw_final).clip(lower=0)).sum() / n_years / 1e3  # MAF/yr
    gw_label = f"{gw_od:.2f} MAF/yr overdraft"

    # ---------------------------------------------------------------------------
    # Shortage by region
    # ---------------------------------------------------------------------------
    sc_ann = sc.resample('YE-SEP').sum() / 1e3   # $M/yr
    sv_ann = sv.resample('YE-SEP').sum()          # TAF/yr

    regions = sorted({c.replace('_urban', '').replace('_ag', '')
                      for c in sc.columns if c.endswith('_urban') or c.endswith('_ag')})

    shortage_region_md_rows = []
    for region in regions:
        sc_u = sc_ann[f'{region}_urban'].mean() if f'{region}_urban' in sc_ann.columns else 0.0
        sc_a = sc_ann[f'{region}_ag'].mean()    if f'{region}_ag'    in sc_ann.columns else 0.0
        sv_u = sv_ann[f'{region}_urban'].mean() if f'{region}_urban' in sv_ann.columns else 0.0
        sv_a = sv_ann[f'{region}_ag'].mean()    if f'{region}_ag'    in sv_ann.columns else 0.0
        shortage_region_md_rows.append(
            f'| {region} | {sc_u:.1f} | {sc_a:.1f} | {sc_u+sc_a:.1f} | {sv_u:.0f} | {sv_a:.0f} | {sv_u+sv_a:.0f} |'
        )

    shortage_region_table = (
        '| Region | Urban cost ($M/yr) | Ag cost ($M/yr) | Total cost ($M/yr)'
        ' | Urban vol (TAF/yr) | Ag vol (TAF/yr) | Total vol (TAF/yr) |\n'
        '|--------|-------------------|-----------------|------------------'
        '|-------------------|-----------------|-------------------|\n'
        + '\n'.join(shortage_region_md_rows)
    )

    # ---------------------------------------------------------------------------
    # Top 10 shortage links by mean annual cost
    # ---------------------------------------------------------------------------
    link_sc = sc[_link_cols(sc)].resample('YE-SEP').sum().mean() / 1e3  # $M/yr
    link_sv = sv[_link_cols(sv)].resample('YE-SEP').sum().mean()        # TAF/yr

    top_links = link_sc.nlargest(10).index
    top_rows  = [
        f'| {lnk} | {link_sc[lnk]:.1f} | {link_sv.get(lnk, 0):.0f} |'
        for lnk in top_links
    ]

    top_links_table = (
        '| Link | Mean annual cost ($M/yr) | Mean annual volume (TAF/yr) |\n'
        '|------|--------------------------|-----------------------------|\n'
        + '\n'.join(top_rows)
    )

    # ---------------------------------------------------------------------------
    # Operation costs by group
    # ---------------------------------------------------------------------------
    col_to_group = op_groups['group'].to_dict()
    group_costs  = {}
    for col in oc.columns:
        grp = col_to_group.get(col, 'other')
        group_costs.setdefault(grp, []).append(col)

    oc_ann = oc.resample('YE-SEP').sum() / 1e3  # $M/yr
    group_totals = {}
    for grp, cols in group_costs.items():
        valid = [c for c in cols if c in oc_ann.columns]
        if valid:
            group_totals[grp] = oc_ann[valid].sum(axis=1).mean()

    oc_group_table = (
        '| Group | Mean annual cost ($M/yr) |\n'
        '|-------|-------------------------|\n'
        + '\n'.join(
            f'| {grp} | {val:.1f} |'
            for grp, val in sorted(group_totals.items(), key=lambda x: -x[1])
        )
    )

    # ---------------------------------------------------------------------------
    # Storage statistics
    # ---------------------------------------------------------------------------
    gw_total = storage[gw_cols].sum(axis=1) / 1e3  # MAF
    sr_total = storage[sr_cols].sum(axis=1) / 1e3  # MAF

    storage_table = (
        '| Component | Mean (MAF) | Min (MAF) | Max (MAF) |\n'
        '|-----------|-----------|----------|----------|\n'
        f'| Surface reservoirs | {sr_total.mean():.1f} | {sr_total.min():.1f} | {sr_total.max():.1f} |\n'
        f'| Groundwater | {gw_total.mean():.1f} | {gw_total.min():.1f} | {gw_total.max():.1f} |'
    )

    # ---------------------------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------------------------
    print('Generating plots...')
    shortage_timeseries(sc, sv,  outpath=os.path.join(fig_dir, 'shortage_timeseries.png'))
    storage_timeseries(storage[sr_cols + gw_cols], outpath=os.path.join(fig_dir, 'storage_timeseries.png'))
    supply_portfolio(flow, portfolio, outpath=os.path.join(fig_dir, 'supply_portfolio.png'))
    print(f'  Figures saved to {fig_dir}/')

    # ---------------------------------------------------------------------------
    # Write report
    # ---------------------------------------------------------------------------
    now = datetime.datetime.now().strftime('%Y-%m-%d')

    report_text = f"""\
# CALVIN Results: {model_name}

*Generated: {now} | Period: {period_start} – {period_end} ({n_years} water years)*

## Summary

| Metric | Value |
|--------|-------|
| Mean annual shortage cost | {mean_sc:.0f} $M/yr |
| Mean annual shortage volume | {mean_sv:.0f} TAF/yr |
| Mean annual operation cost | {mean_oc:.0f} $M/yr |
| Net GW change | {gw_label} |

## Shortage

Annual total shortage cost (bars, left axis) and shortage volume (line, right axis).

![Shortage timeseries](figures/shortage_timeseries.png)

### Shortage by Region

Mean annual shortage cost and volume broken down by region and sector.

{shortage_region_table}

### Top Shortage Links

Top 10 links by mean annual shortage cost.

{top_links_table}

## Storage

Total surface reservoir and groundwater storage over the simulation period.

![Storage timeseries](figures/storage_timeseries.png)

### Storage Statistics

{storage_table}

## Operation Costs

Mean annual operation cost by infrastructure group.

{oc_group_table}

## Supply Portfolio

Mean monthly supply by region and source type.

![Supply portfolio](figures/supply_portfolio.png)
"""

    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f'  Report written to {report_path}')


if __name__ == '__main__':
    generate_report(MODEL_DIR)
