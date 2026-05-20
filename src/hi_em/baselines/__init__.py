"""External baselines for dialogue topic segmentation."""

from hi_em.baselines.greedyseg_delay2 import GreedySegOnlineDelay2
from hi_em.baselines.texttiling_streaming import StreamingTextTiling

__all__ = ["StreamingTextTiling", "GreedySegOnlineDelay2"]
