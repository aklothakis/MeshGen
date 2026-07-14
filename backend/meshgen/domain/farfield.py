"""Flow-aware outer-domain (farfield) sizing.

Turns the body size and freestream conditions into the numbers the topology
builder needs: how far the farfield sits, how many wall-normal layers, and the
first-cell height that resolves the boundary layer to a target ``y+``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..flow.conditions import FlowConditions

__all__ = ["DomainParams"]


@dataclass
class DomainParams:
    """Outer-domain and wall-normal meshing controls."""

    farfield_radius_factor: float = 6.0   # multiples of the body cross-scale
    n_normal: int = 48                    # wall-normal node count
    n_wrap_mult: int = 1                  # refine the wrap direction by this factor
    y_plus: float = 1.0                   # target first-cell y+
    first_cell_height: float | None = None  # override; else derived from y+
    growth_rate_cap: float = 1.25         # advisory near-wall growth cap
    n_stream_blocks: int = 2              # split streamwise into this many blocks
    n_wrap_blocks: int = 1                # split wrap direction into this many blocks
    smoothing_iters: int = 0              # optional elliptic (Winslow) sweeps (0 = off;
                                          # the wall-normal grid is already orthogonal)
    smoothing_omega: float = 0.4          # smoothing under-relaxation factor

    def resolve_first_cell(
        self, flow: FlowConditions | None, body_length: float
    ) -> float:
        """First wall-cell height (absolute)."""
        if self.first_cell_height is not None:
            return self.first_cell_height
        if flow is not None:
            return flow.first_cell_height(self.y_plus)
        # No flow given: fall back to a small fraction of the body length.
        return body_length * 1e-4

    def farfield_radius(self, cross_scale: float, body_length: float) -> float:
        base = max(cross_scale, 0.25 * body_length)
        return self.farfield_radius_factor * base
