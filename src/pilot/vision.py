"""Image sizing for the vision tower.

Qwen3-VL uses ``patch_size=16`` with a 2x2 merge, so one visual token covers
32x32 pixels (Qwen2/2.5-VL used 14 and 28). Passing ``max_pixels`` down to the
processor is version dependent, so instead we resize the PIL image before it
ever reaches the processor: one code path, no keyword arguments that may not
exist.
"""

from __future__ import annotations

import math
from typing import Any

PATCH = 32  # Qwen3-VL: 16 px patch x 2x2 merge


def side_for_tokens(image_tokens: int, patch: int = PATCH) -> int:
    """Longest side (pixels) that keeps the visual token count near the budget."""
    return max(patch, int(math.sqrt(max(image_tokens, 1)) * patch))


def fit_image(image: Any, image_tokens: int = 512, patch: int = PATCH):
    """Downscale a PIL image so it does not exceed the token budget."""
    side = side_for_tokens(image_tokens, patch)
    width, height = image.size
    if max(width, height) <= side:
        return image
    out = image.copy()
    out.thumbnail((side, side))
    return out


def estimate_tokens(width: int, height: int, patch: int = PATCH) -> int:
    return max(1, (width // patch) * (height // patch))
