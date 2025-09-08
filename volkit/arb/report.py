# volkit/arb/report.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


# --------------------------
# Trade leg
# --------------------------

@dataclass
class TradeLeg:
    asset: str                  # "call" | "put" | "bond" | "future"
    side: str                   # "buy" | "sell"
    qty: float
    strike: Optional[float]     # None for non-options
    price: float                # today's PV (0 for futures)
    notional: Optional[float] = None  # for bonds only

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TradeLeg":
        # accept both new ("strike") and legacy ("K") keys
        strike = d.get("strike", d.get("K"))
        return TradeLeg(
            asset=str(d["asset"]),
            side=str(d["side"]),
            qty=float(d.get("qty", 0.0)),
            strike=(None if strike is None else float(strike)),
            price=float(d.get("price", 0.0)),
            notional=(None if d.get("notional") is None else float(d["notional"])),
        )

    def cashflow_today(self) -> float:
        # buy = cash out, sell = cash in
        return (1.0 if self.side == "sell" else -1.0) * float(self.qty) * float(self.price)


# --------------------------
# Trade
# --------------------------

@dataclass
class Trade:
    type: str
    notes: str
    legs: List[TradeLeg] = field(default_factory=list)
    net_cost: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trade":
        legs = [TradeLeg.from_dict(x) for x in d.get("legs", [])]
        return Trade(
            type=str(d.get("type", "")),
            notes=str(d.get("notes", "")),
            legs=legs,
            net_cost=float(d.get("net_cost", 0.0)),
        )

    def cashflow_total(self) -> float:
        return float(sum(L.cashflow_today() for L in self.legs))

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for leg in self.legs:
            rows.append({
                "Asset": leg.asset,
                "Side": leg.side,
                "Qty": leg.qty,
                "Strike": "" if leg.strike is None else leg.strike,
                "Unit Price": leg.price,
                "Notional": "" if leg.notional is None else leg.notional,
                "Cashflow today": leg.cashflow_today(),
            })
        df = pd.DataFrame(
            rows,
            columns=["Asset","Side","Qty","Strike","Unit Price","Notional","Cashflow today"]
        )
        # totals row
        total_cash = df["Cashflow today"].sum(skipna=True)
        total_notional = pd.to_numeric(
            df.loc[df["Asset"] == "bond", "Notional"], errors="coerce"
        ).sum(skipna=True)
        total_row = {
            "Asset": "TOTAL", "Side": "", "Qty": "", "Strike": "", "Unit Price": "",
            "Notional": ("" if abs(total_notional) == 0 else total_notional),
            "Cashflow today": total_cash,
        }
        return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    # small helper for string rendering
    def _table_str(self) -> str:
        df = self.to_dataframe()
        return df.to_string(index=False)

    def __str__(self) -> str:
        head = f"Trade: {self.type}"
        if self.notes:
            head += f" — {self.notes}"
        recon = (
            f"Net cost (reported): {self.net_cost:,.6f} | "
            f"Cashflow today (sum): {self.cashflow_total():,.6f}"
        )
        return f"{head}\n{self._table_str()}\n{recon}"

    __repr__ = __str__


# --------------------------
# Arb report
# --------------------------

@dataclass
class ArbReport:
    ok: bool
    violations: Dict[str, Dict[str, list]]
    trades: List[Trade]
    tolerances: Dict[str, Any]
    enabled_checks: Dict[str, bool]
    meta: Dict[str, Any]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArbReport":
        trades = [Trade.from_dict(t) for t in d.get("trades", [])]
        return ArbReport(
            ok=bool(d.get("ok", False)),
            violations=d.get("violations", {}),
            trades=trades,
            tolerances=d.get("tolerances", {}),
            enabled_checks=d.get("enabled_checks", {}),
            meta=d.get("meta", {}),
        )

    # Pretty printable overview
    def __str__(self) -> str:
        lines: List[str] = []
        for i, tr in enumerate(self.trades, 1):
            lines.append(f"\n#{i} {tr.type}: {tr.notes}")
            lines.append(tr._table_str())
            entry_cash = tr.cashflow_total()  # sum of leg cashflows today
            if entry_cash>=0:
                lines.append(f"Entry cash: {entry_cash:+,.6f} (recieve today)")
            else:
                lines.append(f"Entry cash: {entry_cash:+,.6f} (pay today)")
        return "\n".join(lines)

    __repr__ = __str__

    # Concatenate all trades (each incl. TOTAL row)
    def to_dataframe(self) -> pd.DataFrame:
        frames = []
        for i, tr in enumerate(self.trades, 1):
            df = tr.to_dataframe()
            df.insert(0, "Trade #", i)
            df.insert(1, "Type", tr.type)
            df.insert(2, "Notes", tr.notes)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ = ["TradeLeg", "Trade", "ArbReport"]
