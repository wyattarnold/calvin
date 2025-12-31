# %% [markdown]
# ### Baseline data for COSVF in CALVIN
# 
# This script prepares files necessary to instantiate a Carryover Storage Value Function (COSVF) 
# annual limited foresight model of CALVIN. The links file is the export for the first water year 
# (1922) with debug links added from Calvin Network tools.
# 
# **Output files:**
# 1. `links.csv` - Network for the first water year in the period of analysis
# 2. `cosvf-params.csv` - Penalty parameters table (initial values for EA optimization)
# 3. `r-dict.json` - Dictionary of reservoirs with penalty properties
# 4. `inflows.csv` - External inflows for every monthly time step
# 5. `variable-constraints.csv` - Links with variable year-to-year bounds

# %%
import os
import sys
import json
import numpy as np
import pandas as pd

# %%
# Load CALVIN library
calvin_dir = os.path.abspath('../../')
if str(calvin_dir) not in sys.path:
    sys.path.append(calvin_dir)
from calvin import *

# %%
# Model directories
pf_dir = '../../../my-models/calvin-pf'  # perfect foresight
lf_dir = '../../../my-models/calvin-lf'  # limited foresight

# %%
# Load perfect foresight links
pf_links = pd.read_csv(os.path.join(pf_dir, 'links82yr.csv'))

# %%
# Remove debug nodes
pf_links = pf_links.loc[
    (~pf_links.i.str.contains('DBUG')) & (~pf_links.j.str.contains('DBUG'))]

# %%
# Create columns for edges and temporal information
pf_links.insert(0, 'edge', 
    value=pf_links.i.str.split('.').str[0] + '_' + pf_links.j.str.split('.').str[0])
pf_links.insert(1, 'i_node',
    value=pf_links.i.map(lambda x: x.split('.')[0]))
pf_links.insert(2, 'j_node',
    value=pf_links.j.map(lambda x: x.split('.')[0]))
pf_links.insert(0, 'year',
    value=pd.DatetimeIndex(pf_links.i.str.split('.').str[1]).year)
pf_links.insert(1, 'month',
    value=pd.DatetimeIndex(pf_links.i.str.split('.').str[1]).month)
pf_links.insert(0, 'date',
    value=pd.DatetimeIndex(pf_links.i.str.split('.').str[1]))

# %%
# Extract monthly inflows from network data
inflow_qwry = pf_links.loc[pf_links.i.str.contains('INFLOW')]
inflows = inflow_qwry['j'].str.split('.', expand=True)
inflows.columns = ['j', 'date']
inflows['date'] = pd.DatetimeIndex(inflows['date'])
inflows.set_index('date', inplace=True)
inflows.insert(1, 'flow_taf', value=inflow_qwry['lower_bound'].values)

# %%
# Save inflows to CSV
inflows.to_csv('inflows.csv')

# %%
# ## Extract variable constraints
# Query upper and lower bounds that vary year-to-year
def get_variable_range(links, column):
    """Find links where bounds vary across years."""
    variable_links = links.groupby(['edge', 'month', 'k'])[column].max().subtract(
        links.groupby(['edge', 'month', 'k'])[column].min())
    return pd.DataFrame(variable_links.iloc[np.where(variable_links > 0)])

# %%
# Filter out INFLOW and INITIAL links
variable_links = pf_links.loc[
    (~pf_links.i.str.contains('INFLOW')) &
    (~pf_links.i.str.contains('INITIAL'))]

variable_lb = get_variable_range(variable_links, 'lower_bound')
variable_ub = get_variable_range(variable_links, 'upper_bound')
variable_min_max = variable_ub.join(variable_lb, how='outer')

# Query the constraints that were found to vary
variable_constraints = pf_links.loc[pf_links['edge'].isin(
    variable_min_max.index.get_level_values(0).unique())]

# Process storage variable constraints (non-September months)
variable_constraints_storages = variable_constraints.loc[
    (variable_constraints.i_node == variable_constraints.j_node) &
    (variable_constraints.month != 9)]

variable_constraints_storages_k0 = variable_constraints_storages.loc[
    variable_constraints_storages.k == 0]

variable_constraints_storages_k0ub = variable_constraints_storages.groupby(
    ['i', 'j'], as_index=False)['upper_bound'].sum()

variable_constraints_storages_k0 = variable_constraints_storages_k0.merge(
    variable_constraints_storages_k0ub, on=['i', 'j'], suffixes=('', '_join'))
variable_constraints_storages_k0['upper_bound'] = variable_constraints_storages_k0['upper_bound_join']

# Set constant lower bounds for Shasta and Clair Engle
variable_constraints_storages_k0.loc[
    variable_constraints_storages_k0.edge == 'SR_SHA_SR_SHA', 'lower_bound'] = 650
variable_constraints_storages_k0.loc[
    variable_constraints_storages_k0.edge == 'SR_CLE_SR_CLE', 'lower_bound'] = 500

# Remove storage constraints, then add back processed storage constraints
variable_constraints = variable_constraints.loc[
    variable_constraints.i_node != variable_constraints.j_node]
variable_constraints = pd.concat([variable_constraints, variable_constraints_storages_k0])   

# %%
# Save variable constraints to CSV
variable_constraints = variable_constraints[['date', 'i', 'j', 'k', 'lower_bound', 'upper_bound']]
variable_constraints.to_csv('variable-constraints.csv', index=False)

# %%
# ## Define reservoir types and parameters
# Get list of all reservoirs
r_list = pf_links.loc[pf_links.i_node.str.startswith('INITIAL')].j_node.unique()

# %%
# Type 1: Surface reservoirs with quadratic COSVF penalties (26 reservoirs)
r_type1 = ['SR_BER', 'SR_BUC', 'SR_BUL', 'SR_CLE', 'SR_CLK_INV', 'SR_CMN', 'SR_DNP',
           'SR_EBMUD', 'SR_FOL', 'SR_HTH', 'SR_ISB', 'SR_LL_ENR', 'SR_LVQ', 'SR_MCR',
           'SR_MIL', 'SR_NHG', 'SR_NML', 'SR_ORO', 'SR_PAR', 'SR_PNF', 'SR_RLL_CMB',
           'SR_SHA', 'SR_SNL', 'SR_SFAGG', 'SR_GNT', 'SR_WHI']

# Type 2: Groundwater basins with linear COSVF penalties (23 basins)
r_type2 = ['GW_01', 'GW_02', 'GW_03', 'GW_04', 'GW_05', 'GW_06', 'GW_07',
           'GW_08', 'GW_09', 'GW_10', 'GW_11', 'GW_12', 'GW_13', 'GW_14', 'GW_15',
           'GW_16', 'GW_17', 'GW_18', 'GW_19', 'GW_20', 'GW_21',
           'GW_AV', 'GW_CH', 'GW_EW', 'GW_IM', 'GW_MJ', 'GW_MWD', 'GW_OW', 'GW_SBV', 'GW_SC', 'GW_SD', 'GW_VC']

# Type 2 fixed: Groundwater basins with fixed penalties (9 basins)
# r_type2_fixed = ['GW_MWD', 'GW_OW', 'GW_SBV', 'GW_SC', 'GW_SD', 'GW_VC',
#                  'GW_CH', 'GW_EW', 'GW_IM']

# r_type2_fixed_costs = {'GW_MWD': -659, 'GW_OW': -729, 'GW_SBV': -1013, 'GW_SC': -452, 'GW_SD': -1206,
#                        'GW_VC': -1983, 'GW_CH': -71, 'GW_EW': -1115, 'GW_IM': -1}

# r_type2_fixed_init = {'GW_MWD': 750, 'GW_OW': 30000, 'GW_SBV': 2500, 'GW_SC': 425, 'GW_SD': 7000,
#                       'GW_VC': 275, 'GW_CH': 3500, 'GW_EW': 7000, 'GW_IM': 930}   

# %%
# Build reservoir dictionary for CALVIN limited foresight model
r_dict = {}
i = 0
for r in r_list:
    # Extract reservoir properties from perfect foresight network
    initial_storage = pf_links.loc[
        (pf_links.i_node == 'INITIAL') & (pf_links.j_node == r)].lower_bound
    
    lb_9 = pf_links.loc[
        (pf_links.i_node == r) & (pf_links.j_node == r) &
        (pf_links.k == 0) & (pf_links.month == 9)].lower_bound.min()
    
    ub_9 = pf_links.loc[
        (pf_links.i_node == r) & (pf_links.j_node == r) &
        (pf_links.month == 9) & (pf_links.year == 1922)].upper_bound.sum()
    
    # Assign COSVF type and parameter indices
    if r in r_type1:
        r_type, cosvf_param_index, k_count, i = 1, [i, i+1], 15, i+2
    elif r in r_type2:
        r_type, cosvf_param_index, k_count, i = 2, i, 2, i+1
    else:
        r_type, cosvf_param_index, k_count = 0, None, 1
    
    # Add to reservoir dictionary
    r_dict[r] = {
        'eop_init': initial_storage.values[0],
        'lb': lb_9,
        'ub': ub_9,
        'type': r_type,
        'cosvf_param_index': cosvf_param_index,
        'k_count': k_count
    }

# %%
# Save reservoir dictionary to JSON
with open('r-dict.json', 'w') as json_file:
    json.dump(r_dict, json_file, sort_keys=False, indent=4, separators=(',', ': '))

# %%
# ## Create default COSVF parameters
# Build default COSVF penalty parameters (initial values for EA)
param = ['pmin', 'pmax']
rtype1_list = [key for key, value in r_dict.items() if value['type'] == 1]
rtype2_list = [key for key, value in r_dict.items() if value['type'] == 2]
pos_r_list = rtype2_list + list(np.repeat(rtype1_list, len(param)))

cosvf_pminmax = pd.DataFrame({
    'value': list(np.repeat([-1e2], len(rtype2_list))) + list(np.tile([-1e2, -7e2], len(rtype1_list)))
})
cosvf_pminmax.insert(0, 'r', value=pos_r_list)
cosvf_pminmax.insert(1, 'param', value=['p'] * len(rtype2_list) + param * len(rtype1_list))

# %%
# Save default COSVF parameters
cosvf_pminmax.to_csv(os.path.join(lf_dir, 'cosvf-params.csv'), index=False)

# %% [markdown]

# %%
# Load default links and add node/edge columns
links = pd.read_csv('links_default.csv')
links.insert(0, 'edge',
    value=links.i.str.split('.').str[0] + '_' + links.j.str.split('.').str[0])
links.insert(1, 'i_node',
    value=links.i.map(lambda x: x.split('.')[0]))
links.insert(2, 'j_node',
    value=links.j.map(lambda x: x.split('.')[0]))

# %%
# Process Type 1 surface reservoir storage links
links_storages = links.loc[(links.i_node == links.j_node) & (links.i_node.isin(r_type1))]
links_storages_k0 = links_storages.loc[links_storages.k == 0].copy()

# Set default storage persuasion penalty to -0.02 $/af
links_storages_k0['cost'] = -0.02

# Calculate total upper bound across all k links
links_storages_k0ub = links_storages.groupby(['i', 'j'], as_index=False)['upper_bound'].sum()
links_storages_k0 = links_storages_k0.merge(
    links_storages_k0ub, on=['i', 'j'], suffixes=('', '_join'))
links_storages_k0['upper_bound'] = links_storages_k0['upper_bound_join']

# Set constant lower bounds for key reservoirs
links_storages_k0.loc[links_storages_k0.edge == 'SR_SHA_SR_SHA', 'lower_bound'] = 650
links_storages_k0.loc[links_storages_k0.edge == 'SR_CLE_SR_CLE', 'lower_bound'] = 500

# Remove and replace storage links in main dataframe
r_type1_concat = [x + '_' + x for x in r_type1]
links = links.loc[~links.edge.isin(r_type1_concat)]
links = pd.concat([links, links_storages_k0])   

# %%
# Process Type 2 fixed groundwater storage links
links_gw_storages = links.loc[
    (links.i_node.isin(r_type2_fixed)) & (links.j_node == 'FINAL')].copy()

links_gw_storages['k'] = 1

def apply_ub(df):
    """Calculate available groundwater storage capacity."""
    df.upper_bound = pf_links.loc[
        (pf_links.i_node == df.i_node) & (pf_links.j_node == df.i_node) &
        (pf_links.month == 9) & (pf_links.year == 1922)].upper_bound.sum() - \
        pf_links.loc[(pf_links.i_node == 'INITIAL') & (pf_links.j_node == df.i_node)].lower_bound.values[0]
    return df

links_gw_storages = links_gw_storages.apply(apply_ub, axis=1)
links_gw_storages['lower_bound'] = 0

# %%
# Apply fixed costs and initial values to groundwater links
# mask_gw_final = (links.i_node.isin(r_type2_fixed)) & (links.j_node == 'FINAL')
# links.loc[mask_gw_final, 'cost'] = links.loc[links.j_node == 'FINAL'].i_node.map(r_type2_fixed_costs)
# links.loc[mask_gw_final, 'upper_bound'] = links.loc[links.j_node == 'FINAL'].i_node.map(r_type2_fixed_init)

# Add processed groundwater storage links
links = pd.concat([links, links_gw_storages])
# links.loc[mask_gw_final, 'lower_bound'] = 0

# %%
# Save final links file
links[['i', 'j', 'k', 'cost', 'amplitude', 'lower_bound', 'upper_bound']].to_csv('links.csv', index=False)

