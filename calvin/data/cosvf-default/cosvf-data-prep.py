# %% Prepare necessary COSVF default inputs
import os
import json
import numpy as np
import pandas as pd

from calvin import *

pf_dir = './my-models/calvin-pf'
lf_dir = './my-models/calvin-lf'


#  %% load perfect foresight links
pf_links = pd.read_csv(os.path.join(pf_dir,'links82yr.csv'))
# remove debug nodes
pf_links = pf_links.loc[
    (~pf_links.i.str.contains('DBUG')) & (~pf_links.j.str.contains('DBUG'))]
# create a column for edges (w/o dates)
pf_links.insert(0, 'edge', 
    value=pf_links.i.str.split('.').str[0]+'_'+pf_links.j.str.split('.').str[0])
pf_links.insert(1,'i_node',
    value=pf_links.i.map(lambda x: x.split('.')[0]))
pf_links.insert(2,'j_node',
    value=pf_links.j.map(lambda x: x.split('.')[0]))
pf_links.insert(0, 'year',
    value=pd.DatetimeIndex(pf_links.i.str.split('.').str[1]).year)
pf_links.insert(1, 'month',
    value=pd.DatetimeIndex(pf_links.i.str.split('.').str[1]).month)
pf_links.insert(0, 'date',
    value=pd.DatetimeIndex(pf_links.i.str.split('.').str[1]))


# %% Inflows
# Extract monthly inflows to csv
# load inflows 
inflow_qwry = pf_links.loc[(pf_links.i.str.contains('INFLOW'))]
# split j to node and date
inflows = inflow_qwry['j'].str.split('.',expand=True)
inflows.columns = ['j','date']
inflows['date'] = pd.DatetimeIndex(inflows['date'])
inflows.set_index('date',inplace=True)
# get inflow values
inflows.insert(1,'flow_taf', value = inflow_qwry['lower_bound'].values)

# %%  save out inflows output
inflows.to_csv(os.path.join(lf_dir,'inflows.csv'))


# %% Variable Constraints
# Query out upper and lower bounds that change from year to year and export to csv
def get_variable_range(links,column):
    variable_links = links.groupby(
        ['edge','month','k'])[column].max().subtract(
        links.groupby(['edge','month','k'])[column].min())
    return(pd.DataFrame(variable_links.iloc[np.where(variable_links>0)]))

variable_links = pf_links.loc[
    (~pf_links.i.str.contains('INFLOW')) &
    (~pf_links.i.str.contains('INITIAL')) &
    (~pf_links.j.str.contains('SINK'))]
variable_lb = get_variable_range(variable_links,'lower_bound')
variable_ub = get_variable_range(variable_links,'upper_bound')
variable_min_max = variable_ub.join(variable_lb,how='outer')
variable_min_max.head()
# query the constraints that were found to have vary
variable_constraints = pf_links.loc[pf_links['edge'].isin(
    variable_min_max.index.get_level_values(0).unique())]
# subset storage variable constraints
variable_constraints_storages = variable_constraints.loc[
    (variable_constraints.i_node==variable_constraints.j_node) &
    (variable_constraints.month!=9)]
# remove storage variable constraints from main variable constraints
variable_constraints = variable_constraints.loc[
    variable_constraints.i_node!=variable_constraints.j_node]
# add back in the variable storage constraints for all other months than September
variable_constraints = variable_constraints.append(variable_constraints_storages)   

# %% save out variable constraints output
variable_constraints = variable_constraints[['date','i','j','k','lower_bound','upper_bound']]
variable_constraints.to_csv(
    os.path.join(lf_dir,'variable-constraints.csv'),index=False)


# %% Reservoirs
# List of reservoirs
r_list = pf_links.loc[(pf_links.i_node.str.startswith('INITIAL'))].j_node.unique()

# Reservoirs identified as COSVF canditates (Type 1)
r_type1 = ['SR_BER','SR_BUC','SR_BUL','SR_CLE','SR_CLK_INV','SR_CMN','SR_DNP',
            'SR_EBMUD','SR_FOL','SR_HTH','SR_ISB','SR_LL_ENR','SR_LVQ','SR_MCR',
            'SR_MIL','SR_NHG','SR_NML','SR_ORO','SR_PAR','SR_PNF','SR_RLL_CMB',
            'SR_SHA','SR_SNL','SR_SFAGG','SR_GNT','SR_WHI']

r_type2 = ['GW_01', 'GW_02', 'GW_05', 
           'GW_08', 'GW_10', 'GW_11', 'GW_12', 'GW_13',
           'GW_16', 'GW_17', 'GW_18', 'GW_20', 'GW_21',
           'GW_AV', 'GW_CH', 'GW_EW', 'GW_MJ', 'GW_MWD',
           'GW_OW', 'GW_SBV', 'GW_SC', 'GW_SD', 'GW_VC']

# reservoir dictionary for calvin limited foresight run
r_dict = dict()
i = 0
for r in r_list:
    # initial storage value
    initial_storage = pf_links.loc[
        (pf_links.i_node=='INITIAL') & (pf_links.j_node==r)].lower_bound
    # lower bound on carryover
    lb_9 = pf_links.loc[
        (pf_links.i_node==r) & (pf_links.j_node==r) & 
        (pf_links.k==0) & (pf_links.month==9)].lower_bound.min()
    # upper bound on carryover from first year
    ub_9 = pf_links.loc[
        (pf_links.i_node==r) & (pf_links.j_node==r) & 
        (pf_links.month==9) & (pf_links.year==1922)].upper_bound.sum()
    # check COSVF Type 1 to index COSVF param
    if r in r_type1:
        r_type, cosvf_param_index, k_count, i = 1, [i,i+1], 15, i+2
    elif r in r_type2:
        r_type, cosvf_param_index, k_count, i = 2, i, 2, i+1
    else:
        r_type, cosvf_param_index, k_count = 0, None, 1
    # add to reservoir dictionary
    r_dict[r] = dict([
        ('eop_init',initial_storage.values[0]),
        ('lb',lb_9),
        ('ub',ub_9),
        ('type',r_type), 
        ('cosvf_param_index',cosvf_param_index),
        ('k_count',k_count)])

# %% save out the reservoir dictionary to json file
with open(os.path.join(lf_dir,'r-dict.json'), 'w') as json_file:
    json.dump(r_dict, json_file, 
        sort_keys=False, indent=4, separators=(',', ': '))

# %% Create default COSVF params
param=['pmin','pmax']
rtype1_list = list({key: value for key, value in r_dict.items() if value['type'] == 1}.keys())
rtype2_list = list({key: value for key, value in r_dict.items() if value['type'] == 2}.keys())
pos_r_list = rtype2_list + list(np.repeat(rtype1_list, len(param)))
cosvf_pminmax = pd.DataFrame({'value':
    list(np.repeat([-1e2], len(rtype2_list))) + list(np.tile([-1e2, -7e2], len(rtype1_list)))})
cosvf_pminmax.insert(0,'r',value=pos_r_list)
cosvf_pminmax.insert(1,'param',value=list(['p'] * len(rtype2_list) + param * len(rtype1_list)))

# %% save out default COSVF params
cosvf_pminmax.to_csv(os.path.join(lf_dir,'cosvf-params.csv'),index=False)


