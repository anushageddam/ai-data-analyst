from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .measure_engine import MeasureEngine, MeasureError


class AnalyticsPlanError(ValueError):
    """Raised when a natural-language analytical request cannot be planned safely."""


class AnalyticsPlanner:
    """Deterministic analysis planner: question -> intent -> fields -> calculation -> result."""

    def __init__(self, df: pd.DataFrame, semantic_summary: Dict[str, Any]):
        self.df = df
        self.semantic = semantic_summary.get("semantic_roles", semantic_summary)
        self.measure_engine = MeasureEngine(df)

    def _resolve_field(self, query: str, fields: list[str]) -> Optional[str]:
        q = query.lower()
        for field in sorted(fields, key=len, reverse=True):
            label = str(field).lower().replace("_", " ")
            if label in q or str(field).lower() in q:
                return field
        for field in sorted(fields, key=len, reverse=True):
            tokens = [t for t in re.split(r"[_\s-]+", str(field).lower()) if len(t) >= 4]
            if tokens and any(t in q for t in tokens):
                return field
        return None

    def _resolve_entities(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        measure = self._resolve_field(query, self.semantic.get("all_measures", []))
        dimension = self._resolve_field(query, self.semantic.get("all_dimensions", []))

        if not measure:
            measure = self.semantic.get("primary_measure")
        if not dimension:
            dimension = self.semantic.get("primary_dimension")
        return dimension, measure

    def _intent(self, query: str) -> str:
        q = query.lower()
        if re.search(r"\b(top|bottom|highest|lowest|best|worst)\b", q):
            return "ranking"
        if any(x in q for x in ("trend", "over time", "monthly", "weekly", "daily", "growth")):
            return "trend"
        if any(x in q for x in ("average", "mean", "median", "minimum", "maximum", "max", "min")):
            return "aggregation"
        if any(x in q for x in ("total", "sum", "overall", "how much", "how many")):
            return "aggregation"
        if any(x in q for x in ("show", "breakdown", "by ", "distribution", "compare")):
            return "breakdown"
        return "breakdown"

    def plan(self, query: str) -> Dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            raise AnalyticsPlanError("Please enter an analytical question.")

        dimension, measure = self._resolve_entities(q)
        intent = self._intent(q)

        if not measure or measure not in self.df.columns:
            raise AnalyticsPlanError("I could not identify a valid measure from the current dataset.")
        if intent in {"breakdown", "ranking"} and (not dimension or dimension not in self.df.columns):
            raise AnalyticsPlanError("I could not identify a valid dimension for this analysis.")

        n_match = re.search(r"\b(?:top|bottom)\s*(\d+)\b", q.lower())
        top_n = int(n_match.group(1)) if n_match else 5
        if top_n < 1 or top_n > 1000:
            raise AnalyticsPlanError("Top N must be between 1 and 1000.")

        aggregation = "SUM"
        q_lower = q.lower()
        for word, fn in (
            ("average", "AVERAGE"), ("mean", "AVERAGE"), ("median", "MEDIAN"),
            ("minimum", "MIN"), ("maximum", "MAX"), ("max", "MAX"),
            ("min", "MIN"), ("count", "COUNT"), ("distinct", "DISTINCTCOUNT"),
        ):
            if word in q_lower:
                aggregation = fn
                break

        return {
            "question": q,
            "intent": intent,
            "dimension": dimension,
            "measure": measure,
            "aggregation": aggregation,
            "top_n": top_n if intent == "ranking" else None,
            "time_dimension": self.semantic.get("primary_time"),
        }

    def _group_and_aggregate(self, dimension: str, measure: str, aggregation: str) -> pd.DataFrame:
        grouped = self.df.groupby(dimension, dropna=False)[measure]
        if aggregation == "SUM":
            result = grouped.sum()
        elif aggregation == "AVERAGE":
            result = grouped.mean()
        elif aggregation == "MIN":
            result = grouped.min()
        elif aggregation == "MAX":
            result = grouped.max()
        elif aggregation == "COUNT":
            result = grouped.count()
        elif aggregation == "MEDIAN":
            result = grouped.median()
        elif aggregation == "DISTINCTCOUNT":
            result = grouped.nunique()
        else:
            raise AnalyticsPlanError(f"Unsupported aggregation: {aggregation}")
        return result.reset_index()

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        intent = plan["intent"]
        dimension = plan.get("dimension")
        measure = plan["measure"]
        aggregation = plan.get("aggregation", "SUM")

        if intent == "aggregation":
            value = self.measure_engine.evaluate(aggregation, measure)
            return {"kind": "scalar", "value": round(value, 6)}

        if intent == "ranking":
            grouped = self._group_and_aggregate(dimension, measure, aggregation)
            q_lower = plan["question"].lower()
            ascending = "bottom" in q_lower or any(
                x in q_lower for x in ("lowest", "worst")
            )
            grouped = grouped.sort_values(
                measure, ascending=ascending, kind="mergesort"
            ).head(plan["top_n"]).reset_index(drop=True)
            return {"kind": "table", "data": grouped.to_dict(orient="records")}

        if intent == "trend":
            time_dimension = plan.get("time_dimension")
            if not time_dimension or time_dimension not in self.df.columns:
                raise AnalyticsPlanError("No usable date field was detected for this trend request.")
            temp = self.df[[time_dimension, measure]].copy()
            temp["_date"] = pd.to_datetime(temp[time_dimension], errors="coerce")
            temp = temp.dropna(subset=["_date"])
            if temp.empty:
                raise AnalyticsPlanError("The detected date field contains no usable dates.")
            temp["_period"] = temp["_date"].dt.to_period("M").astype(str)
            trend = temp.groupby("_period")[measure].agg(aggregation.lower()).reset_index()
            return {"kind": "table", "data": trend.to_dict(orient="records"), "time_dimension": time_dimension}

        grouped = self._group_and_aggregate(dimension, measure, aggregation)
        grouped = grouped.sort_values(measure, ascending=False, kind="mergesort").head(100)
        return {"kind": "table", "data": grouped.to_dict(orient="records")}

    def analyze(self, query: str) -> Dict[str, Any]:
        plan = self.plan(query)
        try:
            result = self.execute(plan)
        except (KeyError, TypeError, ValueError, MeasureError) as exc:
            raise AnalyticsPlanError(f"Analysis could not be executed safely: {exc}") from exc

        return {"plan": plan, "result": result, "validated": True}
