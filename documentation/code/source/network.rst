Network Tools API
==================

The ``calvin.network`` package provides Python tools for working with the 
`calvin-network-data <https://github.com/ucd-cws/calvin-network-data>`_ repository. 
It replaces the legacy Node.js ``calvin-network-tools`` (``cnf`` CLI) entirely in Python.

Top-level imports:

.. code-block:: python

   from calvin.network import load_network, build_matrix, export_matrix


loader
-------

Crawls the data repository, reads ``node.geojson`` / ``link.json`` files, resolves 
``$ref`` pointers to CSV data, and returns a ``Network`` object.

.. automodule:: calvin.network.loader
    :members:
    :undoc-members:
    :show-inheritance:


matrix
-------

Builds the time-expanded ``i,j,k,cost,amplitude,lower_bound,upper_bound`` matrix from 
a loaded ``Network`` object.

.. automodule:: calvin.network.matrix
    :members:
    :undoc-members:
    :show-inheritance:


query
------

List and find nodes, links, and regions in the data repository.

.. automodule:: calvin.network.query
    :members:
    :undoc-members:
    :show-inheritance:


validate
---------

Validates data repository structure, schemas, and referential integrity.

.. automodule:: calvin.network.validate
    :members:
    :undoc-members:
    :show-inheritance:


apply_changes
--------------

Import data updates from Excel workbooks or CSV files into the data repository.

.. automodule:: calvin.network.apply_changes
    :members:
    :undoc-members:
    :show-inheritance:


prepare
--------

Prepare input files for the COSVF limited-foresight model.

.. automodule:: calvin.network.prepare
    :members:
    :undoc-members:
    :show-inheritance:


cli
----

Command-line interface for all network tools.

.. automodule:: calvin.network.cli
    :members:
    :undoc-members:
    :show-inheritance:
