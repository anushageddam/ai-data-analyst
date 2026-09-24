import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from backend.core.measure_engine import MeasureEngine, MeasureError


class KPIEngine:
    """
    Automatic KPI Generation Engine for YOO PROJECT.
    Generates meaningful, domain-aware KPIs, composite calculated metrics,
    and period-over-period trend comparisons.
    """

    def __init__(self, df: pd.DataFrame, semantic_summary: Dict[str, Any]):
        self.df = df
        self.semantic = semantic_summary["semantic_roles"]
        self.domain = semantic_summary["detected_domain"]
        self.primary_time = self.semantic.get("primary_time")
        self.measures = MeasureEngine(df)

    def _format_value(self, val: float, metric_type: str = "number", prefix: str = "") -> str:
        """Formats numbers with readable abbreviations (K, M, B, %)."""
        if val is None or math.isnan(val):
            return "N/A"

        abs_val = abs(val)
        if metric_type == "percentage":
            return f"{val:.1f}%"

        if abs_val >= 1_000_000_000:
            formatted = f"{val / 1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000:
            formatted = f"{val / 1_000_000:.2f}M"
        elif abs_val >= 1_000:
            formatted = f"{val / 1_000:.1f}K"
        elif isinstance(val, int) or abs_val >= 10:
            formatted = f"{val:,.0f}"
        else:
            formatted = f"{val:,.2f}"

        return f"{prefix}{formatted}"

    def _sum(self, column: str) -> float:
        return self.measures.evaluate("SUM", column)

    def _average(self, column: str) -> float:
        return self.measures.evaluate("AVERAGE", column)

    def _distinct_count(self, column: str) -> int:
        return int(self.measures.evaluate("DISTINCTCOUNT", column))

    def _compute_period_delta(self, measure_col: str) -> Tuple[Optional[float], Optional[str], List[float]]:
        """
        Splits dataset by primary time dimension into current vs previous period
        and computes delta percentage and sparkline points.
        """
        sparkline = []
        if not self.primary_time or self.primary_time not in self.df.columns:
            if len(self.df) >= 6:
                try:
                    chunks = np.array_split(self.df[measure_col].dropna().values, 6)
                    sparkline = [round(float(c.sum()), 1) for c in chunks if len(c) > 0]
                except Exception:
                    pass
            return None, None, sparkline

        try:
            time_series = pd.to_datetime(self.df[self.primary_time], errors="coerce")
            valid_mask = time_series.notna() & self.df[measure_col].notna()
            if valid_mask.sum() < 4:
                return None, None, sparkline

            sub_df = self.df.loc[valid_mask].copy()
            sub_df["_dt"] = time_series[valid_mask]
            sub_df = sub_df.sort_values("_dt")

            n_bins = min(8, len(sub_df))
            if n_bins >= 3:
                chunks = np.array_split(sub_df[measure_col].values, n_bins)
                sparkline = [round(float(c.sum()), 1) for c in chunks if len(c) > 0]

            mid_idx = len(sub_df) // 2
            prev_sum = MeasureEngine(sub_df.iloc[:mid_idx]).evaluate("SUM", measure_col)
            curr_sum = MeasureEngine(sub_df.iloc[mid_idx:]).evaluate("SUM", measure_col)

            if prev_sum > 0:
                delta_pct = round(MeasureEngine(sub_df).divide(curr_sum - prev_sum, prev_sum) * 100, 1)
                direction = "positive" if delta_pct >= 0 else "negative"
                return delta_pct, direction, sparkline

        except (MeasureError, Exception):
            pass

        return None, None, sparkline

    def generate_kpis(self) -> List[Dict[str, Any]]:
        """Generates 4-6 prioritized, domain-aware KPI cards."""
        kpis = []
        measures = self.semantic.get("all_measures", [])
        dimensions = self.semantic.get("all_dimensions", [])
        identifiers = self.semantic.get("identifiers", [])
        n_rows = len(self.df)

        currency_sym = ""
        col_text = " ".join(self.df.columns).lower()
        if "₹" in col_text or "inr" in col_text:
            currency_sym = "₹"
        elif "$" in col_text or "usd" in col_text:
            currency_sym = "$"

        if "Procurement" in self.domain:
            po_val_col = next(
                (m for m in measures if any(k in m.lower() for k in ["net_value", "po_value", "amount", "spend", "value"])),
                None,
            )
            if po_val_col:
                total_val = self._sum(po_val_col)
                delta, direction, spark = self._compute_period_delta(po_val_col)
                kpis.append({
                    "id": "kpi_total_spend", "title": "Total PO Value",
                    "value": self._format_value(total_val, prefix=currency_sym),
                    "raw_value": total_val, "measure_column": po_val_col,
                    "delta_percentage": delta, "direction": direction, "sparkline": spark,
                    "icon": "shopping-cart", "badge": "Net Spend"
                })

            id_col = next((i for i in identifiers if any(k in i.lower() for k in ["po", "order", "number"])), None)
            po_count = self._distinct_count(id_col) if id_col else n_rows
            kpis.append({
                "id": "kpi_po_count", "title": "Total Purchase Orders",
                "value": f"{po_count:,}", "raw_value": po_count,
                "measure_column": None, "delta_percentage": None, "direction": "neutral",
                "sparkline": [], "icon": "file-text", "badge": "Volume"
            })

            if po_val_col and po_count > 0:
                avg_po = self._sum(po_val_col) / po_count
                kpis.append({
                    "id": "kpi_avg_po", "title": "Average PO Value",
                    "value": self._format_value(avg_po, prefix=currency_sym),
                    "raw_value": avg_po, "measure_column": po_val_col,
                    "delta_percentage": None, "direction": "neutral",
                    "sparkline": [], "icon": "trending-up", "badge": "Efficiency"
                })

            qty_col = next((m for m in measures if any(k in m.lower() for k in ["qty", "quantity", "volume"])), None)
            if qty_col:
                total_qty = self._sum(qty_col)
                kpis.append({
                    "id": "kpi_total_qty", "title": "Total Ordered Quantity",
                    "value": self._format_value(total_qty), "raw_value": total_qty,
                    "measure_column": qty_col, "delta_percentage": None,
                    "direction": "neutral", "sparkline": [], "icon": "box", "badge": "Units"
                })

            vendor_col = next((d for d in dimensions if any(k in d.lower() for k in ["vendor", "supplier"])), None)
            if vendor_col:
                v_count = self._distinct_count(vendor_col)
                kpis.append({
                    "id": "kpi_active_vendors", "title": "Active Vendors",
                    "value": str(v_count), "raw_value": v_count, "measure_column": None,
                    "delta_percentage": None, "direction": "neutral", "sparkline": [],
                    "icon": "users", "badge": "Supply Base"
                })

        elif "Sales" in self.domain:
            sales_col = next((m for m in measures if any(k in m.lower() for k in ["sales", "revenue", "amount"])), None)
            if not sales_col and measures:
                sales_col = measures[0]

            if sales_col:
                total_sales = self._sum(sales_col)
                delta, direction, spark = self._compute_period_delta(sales_col)
                kpis.append({
                    "id": "kpi_total_sales", "title": "Total Revenue",
                    "value": self._format_value(total_sales, prefix=currency_sym),
                    "raw_value": total_sales, "measure_column": sales_col,
                    "delta_percentage": delta, "direction": direction, "sparkline": spark,
                    "icon": "dollar-sign", "badge": "Topline"
                })

            profit_col = next((m for m in measures if "profit" in m.lower()), None)
            if profit_col:
                total_profit = self._sum(profit_col)
                delta_p, dir_p, spark_p = self._compute_period_delta(profit_col)
                kpis.append({
                    "id": "kpi_total_profit", "title": "Total Profit",
                    "value": self._format_value(total_profit, prefix=currency_sym),
                    "raw_value": total_profit, "measure_column": profit_col,
                    "delta_percentage": delta_p, "direction": dir_p, "sparkline": spark_p,
                    "icon": "activity", "badge": "Bottomline"
                })

                if sales_col and total_sales > 0:
                    margin_pct = self.measures.divide(total_profit, total_sales) * 100
                    kpis.append({
                        "id": "kpi_profit_margin", "title": "Profit Margin",
                        "value": self._format_value(margin_pct, metric_type="percentage"),
                        "raw_value": margin_pct, "measure_column": None,
                        "delta_percentage": None,
                        "direction": "neutral", "sparkline": [], "icon": "percent", "badge": "Margin"
                    })

            id_col = next((i for i in identifiers if any(k in i.lower() for k in ["order", "trans", "id"])), None)
            order_count = self._distinct_count(id_col) if id_col else n_rows
            kpis.append({
                "id": "kpi_order_count", "title": "Total Orders",
                "value": f"{order_count:,}", "raw_value": order_count, "measure_column": None,
                "delta_percentage": None, "direction": "neutral", "sparkline": [],
                "icon": "shopping-bag", "badge": "Transactions"
            })

            if sales_col and order_count > 0:
                aov = self._sum(sales_col) / order_count
                kpis.append({
                    "id": "kpi_aov", "title": "Avg Order Value",
                    "value": self._format_value(aov, prefix=currency_sym),
                    "raw_value": aov, "measure_column": sales_col,
                    "delta_percentage": None, "direction": "neutral", "sparkline": [],
                    "icon": "trending-up", "badge": "Basket Size"
                })

            qty_col = next((m for m in measures if any(k in m.lower() for k in ["qty", "quantity", "volume"])), None)
            if qty_col and len(kpis) < 6:
                total_qty = self._sum(qty_col)
                kpis.append({
                    "id": "kpi_total_qty", "title": "Total Units Sold",
                    "value": self._format_value(total_qty), "raw_value": total_qty,
                    "measure_column": qty_col, "delta_percentage": None,
                    "direction": "neutral", "sparkline": [], "icon": "package", "badge": "Volume"
                })

        else:
            primary_m = self.semantic.get("primary_measure")
            if primary_m:
                tot = self._sum(primary_m)
                delta, direction, spark = self._compute_period_delta(primary_m)
                kpis.append({
                    "id": "kpi_primary_total",
                    "title": f"Total {primary_m.replace('_', ' ').title()}",
                    "value": self._format_value(tot, prefix=currency_sym),
                    "raw_value": tot, "measure_column": primary_m,
                    "delta_percentage": delta, "direction": direction, "sparkline": spark,
                    "icon": "bar-chart-2", "badge": "Primary"
                })

                avg = self._average(primary_m)
                kpis.append({
                    "id": "kpi_primary_avg",
                    "title": f"Average {primary_m.replace('_', ' ').title()}",
                    "value": self._format_value(avg, prefix=currency_sym),
                    "raw_value": avg, "measure_column": primary_m,
                    "delta_percentage": None, "direction": "neutral",
                    "sparkline": [], "icon": "hash", "badge": "Mean"
                })

            kpis.append({
                "id": "kpi_record_count", "title": "Total Records",
                "value": f"{n_rows:,}", "raw_value": n_rows, "measure_column": None,
                "delta_percentage": None, "direction": "neutral", "sparkline": [],
                "icon": "database", "badge": "Entries"
            })

            sec_m = self.semantic.get("secondary_measures", [])
            for m in sec_m[:2]:
                if len(kpis) < 5:
                    tot_s = self._sum(m)
                    kpis.append({
                        "id": f"kpi_{m}", "title": f"Total {m.replace('_', ' ').title()}",
                        "value": self._format_value(tot_s, prefix=currency_sym),
                        "raw_value": tot_s, "measure_column": m,
                        "delta_percentage": None, "direction": "neutral", "sparkline": [],
                        "icon": "layers", "badge": "Measure"
                    })

        return kpis[:6]
