"""
FastAPI application factory for the Calvin Network App.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from calvin.app.routers import network as network_router
from calvin.app.routers import results as results_router
from calvin.app.state import state

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    data_path: str | Path | None = None,
    study_paths: list[str | Path] | None = None,
    default_study: str | None = None,
    prebuilt_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    data_path:
        Path to the calvin-network-data/data directory. Required unless prebuilt_dir is set.
    study_paths:
        List of model run directories (each should contain a results/ subdirectory).
    default_study:
        Name of the study to make active by default. If None, the first study is used.
    prebuilt_dir:
        If set, load pre-built network.geojson + network.json from this directory instead
        of calling load_network() (hosted mode — avoids reading ~7 000 raw files at startup).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if prebuilt_dir is not None:
            state.load_prebuilt(prebuilt_dir)
        else:
            state.load(data_path, study_paths or [], default_study)
        yield
        # (no teardown needed for in-memory state)

    app = FastAPI(
        title="Calvin Network App",
        description="Interactive visualization for the CALVIN water network model.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(network_router.router, prefix="/api/network", tags=["Network"])
    app.include_router(results_router.router, prefix="/api/results", tags=["Results"])

    # Serve the built React frontend if the static directory exists
    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    else:
        logger.warning(
            "Static frontend not found at %s. "
            "Run `cd calvin/app/frontend && npm run build` to build it.",
            _STATIC_DIR,
        )

    return app
