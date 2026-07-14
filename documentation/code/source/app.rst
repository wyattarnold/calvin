Web App
========

CALVIN includes a FastAPI + React web app for interactively exploring the California water
network and optimization results. It can be run locally or accessed at the hosted deployment
at `calvin-network-app.onrender.com <https://calvin-network-app.onrender.com>`_.


Installation
-------------

.. code-block:: bash

   pip install "calvin[app]"


Serve Modes
------------

The app has three mutually exclusive serve modes.

**Hosted** — loads a pre-built network and bundled results from ``data.zip`` (no local data
required). This is how the public deployment runs on Render:

.. code-block:: bash

   python -m calvin.app serve --hosted

**Local** — reads network data from a local ``calvin-network-data`` clone and auto-discovers
model runs under ``./my-models/``:

.. code-block:: bash

   python -m calvin.app serve --data ../calvin-network-data/data --local

Any subdirectory of ``my-models/`` that contains a ``results/`` folder is loaded as a study.

**Explicit** — specify the data path and one or more study directories directly:

.. code-block:: bash

   python -m calvin.app serve \
       --data ../calvin-network-data/data \
       --study my-models/calvin-pf \
       --study my-models/calvin-cosvf


Bundling for Deployment
------------------------

To create a self-contained ``data.zip`` for hosted deployment:

.. code-block:: bash

   python -m calvin.app bundle \
       --data ../calvin-network-data/data \
       --study my-models/calvin-pf

This pre-builds the network JSON and packages result CSVs into ``calvin/app/data.zip``.
Commit the zip to the repository and deploy with ``--hosted``.


Render Deployment
------------------

The app is configured for deployment on `Render <https://render.com>`_ via ``render.yaml``
at the repository root. The build script is ``calvin/app/build.sh``, which installs Python
dependencies and builds the React frontend.

The start command is:

.. code-block:: bash

   python -m calvin.app serve --hosted --host 0.0.0.0 --port $PORT

Render injects the ``$PORT`` environment variable automatically.


Frontend Development
---------------------

The React frontend lives in ``calvin/app/frontend/``. To run the dev server with hot reload
(proxies ``/api`` requests to the FastAPI backend on port 8000):

.. code-block:: bash

   # Terminal 1 — FastAPI backend
   python -m calvin.app serve --data ../calvin-network-data/data --local

   # Terminal 2 — Vite dev server
   cd calvin/app/frontend
   npm install
   npm run dev

To build the frontend for production:

.. code-block:: bash

   cd calvin/app/frontend
   npm run build


API Reference
--------------

server
~~~~~~~

.. automodule:: calvin.app.server
    :members:
    :undoc-members:
    :show-inheritance:

state
~~~~~~

.. automodule:: calvin.app.state
    :members:
    :undoc-members:
    :show-inheritance:

schemas
~~~~~~~~

.. automodule:: calvin.app.schemas
    :members:
    :undoc-members:
    :show-inheritance:

routers.network
~~~~~~~~~~~~~~~~

.. automodule:: calvin.app.routers.network
    :members:
    :undoc-members:
    :show-inheritance:

routers.results
~~~~~~~~~~~~~~~~

.. automodule:: calvin.app.routers.results
    :members:
    :undoc-members:
    :show-inheritance:
