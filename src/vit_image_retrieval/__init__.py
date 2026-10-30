from vit_image_retrieval.core.feature_extractor import ImageFeatureExtractor
from vit_image_retrieval.core.retrieval_system import ImageRetrievalSystem

# Only import main if PyQt5 is available (for desktop GUI)
try:
    from vit_image_retrieval.main import main
    _has_gui = True
except ImportError:
    main = None
    _has_gui = False

__version__ = "1.2.0"
__all__ = ['ImageFeatureExtractor', 'ImageRetrievalSystem', 'main', '_has_gui']