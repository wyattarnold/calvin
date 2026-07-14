Data Files
===========

The ``calvin/data/`` directory contains reference data used by the model during 
postprocessing, supply portfolio analysis, and limited-foresight runs. These files are 
bundled with the package and are **not** user-generated.


SR_stats.csv
-------------

Summary statistics for surface reservoir storage, derived from historical model runs.
Used by ``CALVIN.eop_constraint_multiplier()`` to set end-of-period storage bounds as a 
fraction of capacity, and by ``COSVF`` to validate hydropower lower bounds.

Schema: ``<reservoir>, count, mean, std, min, 25%, 50%, 75%, max``

The ``min`` and ``max`` columns are the key fields — they define :math:`S_{min}` and 
:math:`S_{max}` for each reservoir.


demand_nodes.csv
-----------------

Maps demand links to regions and demand types (urban or agricultural). Used by the 
postprocessor to compute shortage volumes and costs by region.

Schema: ``link, region, type``


portfolio.csv
--------------

Supply portfolio classification for stacked bar chart visualization. Maps each supply link 
to a demand type (ag/urban), supply type (e.g. GWP, SWP, CVP), region, and aggregation 
label.

Schema: ``link, type, supplytype, region, aggreg``


operation_nodes.csv
--------------------

List of links that carry operational costs (e.g. water treatment plants, pump-conveyance). 
Used by the postprocessor to separate operational costs from shortage costs.

Schema: ``link``


operation_groups.csv
---------------------

Groups operational cost links into named categories for aggregated reporting.

Schema: ``link, group``


pwp_nodes.csv
--------------

List of pumping/power plant links. Used for hydropower-related postprocessing.

Schema: ``link``


fourth-assessment-data/
------------------------

Rim inflow multiplier data generated for the California Fourth Climate Assessment (2018).

- ``rim-inflow-overview.xlsx`` — documents which rim inflows have multipliers derived 
  directly from projection data vs. found by correlation.
- ``rim-inflow-multipliers/`` — CSV files for each bias-corrected downscaled GCM 
  projection, with one row per month and one column per reservoir inflow point.
