# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.46'

from .postprocessing.gsi import gsi
from .tracker_zoo import create_tracker, get_tracker_config
from .trackers.botsort.bot_sort import BoTSORT
from .trackers.bytetrack.byte_tracker import BYTETracker
from .trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from .trackers.hybridsort.hybridsort import HybridSORT
from .trackers.ocsort.ocsort import OCSort as OCSORT
from .trackers.strongsort.strong_sort import StrongSORT

TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort']

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT", "HybridSORT",
           "create_tracker", "get_tracker_config", "gsi")
