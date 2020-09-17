### Baseline data for COSVF in CALVIN

The files included here provide the inputs necessary to instatiate a Carryover Storage Value Function (COSVF) annual limited foresight model of CALVIN. The data is prepared based on the [California-Network-Data](https://github.com/ucd-cws/calvin-network-data) commit [b8d35f8...](https://github.com/ucd-cws/calvin-network-data/commit/b8d35f80b6f31e6f2014cdfedaa81d4d85070b07) on Feb 15, 2017. See [cosvf-data-prep.py](cosvf-data-prep.py) for how these files were generated. The links file is the export for the first water year (1922) with debug links added from Calvin Network tools.

**These files automatically pulled into the COSVF model run if the user does not specify their own COSVF model working directory when instatiating a COSVF CALVIN model.**

1. ``links.csv``: 


    Network for the first water year in the period of analysis.
  
    CSV file with column headers: ``i,j,k,cost,amplitude,lower_bound,upper_bound``

2. ``cosvf-params.csv`` 
    
    A table of minimum and maximum marginal cost penalties for quadratic carryover penalty curves on surface water reservoirs and linear penalties on groundwater reservoirs. **The values are not optimized. They are just stand-in values to be replaced by the evolutionary optimization run.**

    CSV file with column headers: ``r,param,value``

3.  ``r-dict.json``: 
    
    A dictionary of reservoirs in the network with penalty properties. 

    Type 2 (linear penalty) reservoirs must be ordered prior to the Type 1 (quadratic) reservoirs. This is a limitation of imposed by the evolutionary algorithm search of the parameters. For each reservoir with an EOP penalty, an index attribute ``cosvf_param_index`` points to the row-index (pythonic zero-indexed) of the list of reservoirs in ``cosvf-params.csv``.

    Dictionary structure:

        {"<reservoir id (e.g. SR_DNP)>":
        
            {
              "eop_init": "(float) initial (October 1) storage level",
              
              "lb": "(float) minimum (end-of-September) storage level",
              
              "ub": "(float) maximum (end-of-September) carryover capactiy",
              
              "type": "(int) 0:none; 1:quadratic; or 2:linear",
              
              "cosvf_param_index": "(list) index to cosvf_params.csv row (pythonic zero-indexed)",
              
              "k_count": "(int) number of piecewise links to fit for quadratic COSVF",
            }

        }

4.  ``inflows.csv``: 
    
    External inflows for every monthly time step over the period of analysis to run. 

    CSV file with column headers ``date, j, flow_taf``

5.  ``variable-constraints.csv``: 
    
    Links that have variable year-to-year upper and/or lower bounds.

    CSV file with column headers ``date,i,j,k,lower_bound,upper_bound``

