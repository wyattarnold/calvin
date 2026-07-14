"""
Standard plots for CALVIN model results.

All functions accept DataFrames loaded from postprocessor output CSVs and an
optional ``outpath`` to save the figure. They return the matplotlib Figure so
callers can further customize or display it.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_clustered_stacked(dfall, labels=None, title="Water Supply Portfolio", H="/", **kwargs):
    """
    Clustered stacked bar chart.

    Each element of ``dfall`` becomes a cluster group (e.g. urban, ag).
    Columns within each DataFrame are the stacked components (supply types).
    The index is the x-axis categories (regions).
    Group bars are offset side-by-side; ag bars get a hatch pattern.

    :param dfall: list of DataFrames (same index and columns)
    :param labels: list of group labels for the hatch legend
    :param title: chart title
    :param H: hatch character (repeated per group)
    :returns: matplotlib Axes
    """
    n_df  = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe   = plt.subplot(111)

    for df in dfall:
        axe = df.plot(kind="bar", linewidth=0, stacked=True,
                      ax=axe, legend=False, grid=False, **kwargs)

    # Each group gets bar_width of horizontal space; groups are offset by that amount.
    bar_width = 1.0 / (n_df + 1)

    h, l = axe.get_legend_handles_labels()
    for i in range(0, n_df * n_col, n_col):
        group_idx = i // n_col
        for pa in h[i:i + n_col]:
            for rect in pa.patches:
                rect.set_x(rect.get_x() + bar_width * group_idx)
                rect.set_hatch(H * group_idx)
                rect.set_width(bar_width)

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + bar_width) / 2.)
    axe.set_xticklabels(dfall[0].index, rotation=0)
    axe.set_title(title)

    n = [axe.bar(0, 0, color="gray", hatch=H * i) for i in range(n_df)]
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    return axe


def shortage_timeseries(shortage_cost, shortage_volume, outpath=None):
    """
    Annual shortage cost (bars, left axis) and volume (line, right axis).

    :param shortage_cost: DataFrame (monthly, index=date)
    :param shortage_volume: DataFrame (monthly, index=date)
    :param outpath: (str) path to save PNG
    :returns: matplotlib Figure
    """
    # exclude regional aggregate columns (e.g. SC_urban, TB_ag) to avoid double-counting
    def _link_cols(df):
        return [c for c in df.columns if not (c.endswith('_urban') or c.endswith('_ag'))]

    sc_annual = shortage_cost[_link_cols(shortage_cost)].sum(axis=1).resample('YE-SEP').sum() / 1e3  # $M/yr
    sv_annual = shortage_volume[_link_cols(shortage_volume)].sum(axis=1).resample('YE-SEP').sum()    # TAF/yr
    years = sc_annual.index.year

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    ax1.bar(years, sc_annual.values, color='steelblue', alpha=0.7, label='Shortage cost')
    ax2.plot(years, sv_annual.values, color='firebrick', linewidth=1.5, label='Shortage volume')

    ax1.set_xlabel('Water Year')
    ax1.set_ylabel('Shortage Cost ($M/yr)')
    ax2.set_ylabel('Shortage Volume (TAF/yr)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    return fig


def storage_timeseries(storage, outpath=None):
    """
    Total surface reservoir and groundwater storage timeseries.

    :param storage: DataFrame (monthly, columns prefixed GW_ or SR_)
    :param outpath: (str) path to save PNG
    :returns: matplotlib Figure
    """
    gw_cols = [c for c in storage.columns if c.startswith('GW_')]
    sr_cols = [c for c in storage.columns if c.startswith('SR_')]

    gw_total = storage[gw_cols].sum(axis=1) / 1e3  # MAF
    sr_total = storage[sr_cols].sum(axis=1) / 1e3  # MAF

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(sr_total.index, sr_total.values, color='steelblue', linewidth=0.8)
    axes[0].set_ylabel('Storage (MAF)')
    axes[0].set_title('Total Surface Reservoir Storage')

    axes[1].plot(gw_total.index, gw_total.values, color='saddlebrown', linewidth=0.8)
    axes[1].set_ylabel('Storage (MAF)')
    axes[1].set_title('Total Groundwater Storage')

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    return fig


def supply_portfolio(flow, portfolio, outpath=None):
    """
    Clustered stacked bar chart of mean monthly water supply by region and source type.

    Urban bars are solid; agricultural bars are hatched. Expects ``flow`` to contain
    regional aggregate columns in the form ``{region}_{supplytype}_{type}``
    (e.g. ``USV_GWP_urban``) as written by :func:`calvin.postprocessor.aggregate_regions`.

    :param flow: DataFrame (monthly flows with regional aggregate columns)
    :param portfolio: DataFrame (portfolio.csv, columns: type, supplytype, region)
    :param outpath: (str) path to save PNG
    :returns: matplotlib Figure
    """
    def _mean_cols(supply_type):
        """Mean monthly flow for each region×supplytype aggregate column."""
        regions     = portfolio.region.unique()
        supplytypes = portfolio.supplytype.unique()
        cols = [f'{P}_{k}_{supply_type}' for P in regions for k in supplytypes]
        means = flow.reindex(columns=cols, fill_value=0.0).mean()
        # reshape to (regions × supplytypes)
        return means.values.reshape(len(regions), len(supplytypes)), list(regions), list(supplytypes)

    urban_vals, regions, supplytypes = _mean_cols('urban')
    ag_vals,    _,       _           = _mean_cols('ag')
    urban_df = pd.DataFrame(urban_vals, index=regions, columns=supplytypes)
    ag_df    = pd.DataFrame(ag_vals,    index=regions, columns=supplytypes)

    fig = plt.figure(figsize=(14, 6))
    ax  = plot_clustered_stacked([urban_df, ag_df], labels=['urban', 'ag'],
                                 title='Water Supply Portfolio')
    ax.set_ylabel('Mean Monthly Flow (TAF/month)')

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    return fig
