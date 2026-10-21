"""Optional local dependencies; normal installations use site-packages."""
from pathlib import Path
import os
import sys


def configure_dependencies():
    custom = os.environ.get('VORONOI_DEPS')
    location = Path(custom) if custom else Path(__file__).resolve().parent/'deps'
    if custom and not location.is_dir():
        raise FileNotFoundError(f'VORONOI_DEPS does not exist: {location}')
    if location.is_dir() and str(location) not in sys.path:
        sys.path.insert(0, str(location))
