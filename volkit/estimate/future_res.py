from dataclasses import dataclass


@dataclass(frozen=True)
class ImpliedFutureResult:  # pragma: no cover
    """Compact result for implied forward/discount from quotes."""

    F: float  # point forward (mid-band)
    F_bid: float  # lower bound on forward
    F_ask: float  # upper bound on forward
    D: float  # point discount factor
    D_min: float  # lower bound on discount factor (D = e^{-rT})
    D_max: float  # upper bound on discount factor

    def __post_init__(self):
        # Ensure nice printing: force built-in floats (not numpy scalars)
        object.__setattr__(self, "F", float(self.F))
        object.__setattr__(self, "F_bid", float(self.F_bid))
        object.__setattr__(self, "F_ask", float(self.F_ask))
        object.__setattr__(self, "D", float(self.D))
        object.__setattr__(self, "D_min", float(self.D_min))
        object.__setattr__(self, "D_max", float(self.D_max))

    def __repr__(self) -> str:
        # Friendly, compact default representation
        return (
            f"ImpliedFutureResult("
            f"F={self.F:.4f}, "
            f"F_bid={self.F_bid:.4f}, F_ask={self.F_ask:.4f}, "
            f"D={self.D:.4f}, "
            f"D_min={self.D_min:.6f}, D_max={self.D_max:.6f})"
        )

    # Optional: rich HTML in notebooks
    def _repr_html_(self) -> str:
        return (
            "<table style='border-collapse:collapse'>"
            "<tr><th style='text-align:left;padding-right:8px'>F</th>"
            f"<td>{self.F:.6f}</td></tr>"
            "<tr><th style='text-align:left;padding-right:8px'>F_bid</th>"
            f"<td>{self.F_bid:.6f}</td></tr>"
            "<tr><th style='text-align:left;padding-right:8px'>F_ask</th>"
            f"<td>{self.F_ask:.6f}</td></tr>"
            "<tr><th style='text-align:left;padding-right:8px'>D</th>"
            f"<td>{self.D:.6f}</td></tr>"
            "<tr><th style='text-align:left;padding-right:8px'>D_min</th>"
            f"<td>{self.D_min:.8f}</td></tr>"
            "<tr><th style='text-align:left;padding-right:8px'>D_max</th>"
            f"<td>{self.D_max:.8f}</td></tr>"
            "</table>"
        )

    # Optional convenience
    def to_dict(self) -> dict:
        return dict(
            F=self.F,
            F_bid=self.F_bid,
            F_ask=self.F_ask,
            D=self.D,
            D_min=self.D_min,
            D_max=self.D_max,
        )
