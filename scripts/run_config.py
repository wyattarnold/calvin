"""
Run configuration loader.

Each run script has a matching default TOML in scripts/configs/.
On first run the default is copied to the model directory so the user
can customise it per-experiment without touching the committed defaults.
Subsequent runs read the model-dir copy.

Usage (from a run script):
    from run_config import load_config
    cfg = load_config('calvin-pf', MODEL_DIR)
    SOLVER = cfg['run']['solver']
    NODE_LB_OVERRIDES = cfg['network'].get('node_lb_overrides', {})
"""
import shutil
import tomllib
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parent / 'configs'


def load_config(script_name: str, model_dir: str | Path) -> dict:
    """Load run config for *script_name*, seeding model_dir on first run.

    Parameters
    ----------
    script_name : str
        Base name of the run script without extension (e.g. ``'calvin-pf'``).
        Must match a file in ``scripts/configs/<script_name>.toml``.
    model_dir : str or Path
        Model output directory.  If ``config.toml`` is absent here, the
        default from ``scripts/configs/`` is copied in before loading.

    Returns
    -------
    dict
        Parsed TOML contents.
    """
    default_toml = _CONFIGS_DIR / f'{script_name}.toml'
    if not default_toml.exists():
        raise FileNotFoundError(
            f'No default config found at {default_toml}. '
            f'Create scripts/configs/{script_name}.toml first.'
        )

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_toml = model_dir / 'config.toml'

    if not model_toml.exists():
        shutil.copy(default_toml, model_toml)
        print(f'Config: copied default to {model_toml}')

    with open(model_toml, 'rb') as f:
        return tomllib.load(f)
