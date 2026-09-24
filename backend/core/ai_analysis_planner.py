from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd

from .analytics_planner import AnalyticsPlanner, AnalyticsPlanError


class AIAnalysisPlanner:
    """
    Natural-language analysis planning layer.

    This layer converts a user question into a structured, validated plan.
    It does not perform calculations and never invents data. Execution is
    delegated to the deterministic AnalyticsPlanner.
    """

    def __init__(self, df: pd.DataFrame, semantic_summary: Dict[str, Any]):
        self.df = df
        self.semantic_summary = semantic_summary
        self.analytics_planner = AnalyticsPlanner(df, semantic_summary)

    def _extract_filters(self, question: str, dimension: Optional[str]) -> list[Dict[str, Any]]:
        if not dimension or dimension not in self.df.columns:
            return []

        q = question.lower()
        filters = []

        # Only accept an explicit "dimension is/value" style filter when the
        # value exists in the active dataset. This prevents invented filters.
        pattern = rf"\b{re.escape(str(dimension).lower().replace('_', ' '))}\s*(?:is|=|equals|equal to)\s+([^,;]+)"
        match = re.search(pattern, q)
        if not match:
            return filters

        raw_value = match.group(1).strip().strip("'\"")
        values = self.df[dimension].dropna().astype(str)
        exact = next((v for v in values.unique() if v.lower() == raw_value.lower()), None)
        if exact is not None:
            filters.append({"field": dimension, "operator": "equals", "value": exact})

        return filters

    def plan(self, question: str) -> Dict[str, Any]:
        q = str(question or "").strip()
        if not q:
            raise AnalyticsPlanError("Please enter an analytical question.")

        base = self.analytics_planner.plan(q)
        filters = self._extract_filters(q, base.get("dimension"))

        return {
            "question": q,
            "intent": base["intent"],
            "context": {
                "dataset_rows": int(len(self.df)),
                "dataset_columns": int(len(self.df.columns)),
                "domain": self.semantic_summary.get("detected_domain"),
                "active_filters": filters,
            },
            "required_fields": {
                "dimension": base.get("dimension"),
                "measure": base.get("measure"),
                "time_dimension": base.get("time_dimension"),
            },
            "calculation": {
                "aggregation": base.get("aggregation"),
                "top_n": base.get("top_n"),
            },
            "validation": {
                "measure_exists": bool(base.get("measure") in self.df.columns),
                "dimension_exists": bool(
                    not base.get("dimension") or base.get("dimension") in self.df.columns
                ),
            },
            "execution": {
                "engine": "AnalyticsPlanner",
                "deterministic": True,
            },
            "base_plan": base,
        }

    def analyze(self, question: str) -> Dict[str, Any]:
        plan = self.plan(question)
        result = self.analytics_planner.execute(plan["base_plan"])

        return {
            "plan": plan,
            "result": result,
            "validated": True,
            "evidence": {
                "source": "active_dataframe",
                "rows_used": int(len(self.df)),
            },
        }
