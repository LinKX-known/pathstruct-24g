"""Image sizing for the vision tower.

Qwen-VL tokenises an image into roughly ``(H/28) * (W/28)`` visual tokens after
the 2x2 merge. Passing ``max_pixels`` down to the processor is version
dependent, so instead we resize the PIL image before it ever reaches the
processor: one code path, no keyword arguments that may not exist.
"""

from __future__ import annotations

import math
from typing import Any


def side_for_tokens(image_tokens: int, patch: int = 28) -> int:
    """Longest side (pixels) that keeps the visual token count near the budget."""
    return max(patch, int(math.sqrt(max(image_tokens, 1)) * patch))


def fit_image(image: Any, image_tokens: int = 512, patch: int = 28):
    """Downscale a PIL image so it does not exceed the token budget."""
    side = side_for_tokens(image_tokens, patch)
    width, height = image.size
    if max(width, height) <= side:
        return image
    out = image.copy()
    out.thumbnail((side, side))
    return out


def estimate_tokens(width: int, height: int, patch: int = 28) -> int:
    return max(1, (width // patch) * (height // patch))
