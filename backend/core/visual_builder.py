from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


class VisualizationError(ValueError):
    """Raised when a visual configuration is invalid."""


class VisualBuilder:
    """Deterministic builder for editable BI-style categorical and analytical visuals."""

    CHART_TYPES = {
        "bar", "column", "stacked_bar", "stacked_column",
        "line", "area", "pie", "donut", "scatter", "histogram", "table"
    }
    AGGREGATIONS = {
        "sum", "average", "min", "max", "count",
        "distinct_count", "median"
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _field(self, name: Optional[str], required: bool = True) -> None:
        if required and not name:
            raise VisualizationError("A field must be selected.")
        if name and name not in self.df.columns:
            raise VisualizationError(f"Field '{name}' does not exist in the current dataset.")

    def build(
        self,
        chart_type: str,
        dimension: Optional[str] = None,
        measure: Optional[str] = None,
        aggregation: str = "sum",
        legend: Optional[str] = None,
        top_n: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chart = chart_type.strip().lower()
        agg = aggregation.strip().lower()

        if chart not in self.CHART_TYPES:
            raise VisualizationError(f"Unsupported chart type '{chart_type}'.")
        if agg not in self.AGGREGATIONS:
            raise VisualizationError(f"Unsupported aggregation '{aggregation}'.")

        self._field(dimension, chart not in {"histogram", "table"})
        self._field(measure, chart not in {"table"})
        self._field(legend, False)

        work = self.df.copy()
        for field, expected in (filters or {}).items():
            self._field(field)
            work = work[work[field].isin(expected)] if isinstance(expected, (list, tuple, set)) else work[work[field] == expected]

        if work.empty:
            raise VisualizationError("The selected filters return no rows.")

        if chart == "table":
            columns = [c for c in [dimension, measure, legend] if c] or list(work.columns[:10])
            return {
                "chart_type": "table",
                "columns": columns,
                "data": work[columns].to_dict("records"),
            }

        if chart == "histogram":
            values = pd.to_numeric(work[measure], errors="coerce").dropna()
            if values.empty:
                raise VisualizationError(f"Field '{measure}' must contain numeric values.")
            return {
                "chart_type": "histogram",
                "field": measure,
                "data": values.tolist(),
            }

        if chart == "scatter":
            x = pd.to_numeric(work[dimension], errors="coerce")
            y = pd.to_numeric(work[measure], errors="coerce")
            points = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(points) < 2:
                raise VisualizationError("Scatter plots require at least two numeric observations.")
            return {
                "chart_type": "scatter",
                "x_field": dimension,
                "y_field": measure,
                "data": points.to_dict("records"),
            }

        keys = [dimension] + ([legend] if legend else [])
        if agg == "count":
            grouped = work.groupby(keys, dropna=False)[measure].count()
        elif agg == "distinct_count":
            grouped = work.groupby(keys, dropna=False)[measure].nunique()
        else:
            numeric = pd.to_numeric(work[measure], errors="coerce")
            temp = work.copy()
            temp[measure] = numeric
            grouped = temp.groupby(keys, dropna=False)[measure].agg(agg)

        result = grouped.reset_index(name="value").dropna(subset=["value"])
        result = result.sort_values("value", ascending=False)

        if top_n is not None:
            if top_n < 1:
                raise VisualizationError("Top N must be a positive integer.")
            result = result.head(top_n)

        return {
            "chart_type": chart,
            "dimension": dimension,
            "measure": measure,
            "aggregation": agg,
            "legend": legend,
            "data": result.to_dict("records"),
        }
