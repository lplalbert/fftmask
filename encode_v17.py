"""v17 watermark encoder compatibility layer.

The repository's v17 training configuration uses two payload rings and no
rotation ring.  WatermarkV11 contains the same payload construction, so this
class reuses it and disables the optional rotation pattern.
"""

import numpy as np

from encode_v11 import WatermarkV11


class WatermarkV17(WatermarkV11):
    """Generate the two-ring, 60-bit v17 spatial template."""

    def __init__(self, L1=512, k1=30000, r_watermark=None, bitsf=None,
                 r_range=1, n_sectors=60):
        super().__init__(
            L1=L1,
            k1=k1,
            r_watermark=r_watermark or [8, 15],
            bitsf=bitsf or [20, 40],
            r_rotation=None,
            r_range=r_range,
            n_sectors=n_sectors,
        )

    def generate_rotation_pattern(self):
        """Return no rotation ring; v17 has payload rings only."""
        return np.zeros((self.L1, self.L1), dtype=np.float32)
