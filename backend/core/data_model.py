"""Analytical data-model layer for YOO PROJECT.

Builds a serializable model from profiler + semantic results without changing
the underlying dataset. This is the shared contract for future measures,
visuals, filters and AI planning.
"""

from typing import Any, Dict, List

import pandas as pd


class AnalyticalModel:
    """Build a table/field model for the active dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        profiler_output: Dict[str, Any],
        semantic_summary: Dict[str, Any],
    ):
        self.df = df
        self.profiler_output = profiler_output
        self.semantic_summary = semantic_summary
        self._column_profiles = {
            item["name"]: item for item in profiler_output.get("columns", [])
        }

    def build(self) -> Dict[str, Any]:
        roles = self.semantic_summary.get("semantic_roles", {})
        measures = set(roles.get("all_measures", []))
        dimensions = set(roles.get("all_dimensions", []))
        dates = set(roles.get("time_dimensions", []))
        identifiers = set(roles.get("identifiers", []))

        fields: List[Dict[str, Any]] = []

        for name in self.df.columns:
            profile = self._column_profiles.get(str(name), {})

            if name in measures:
                role = "Measure"
            elif name in dates:
                role = "Date"
            elif name in identifiers:
                role = "Identifier"
            elif name in dimensions:
                role = "Dimension"
            else:
                role = "Other"

            fields.append(
                {
                    "name": str(name),
                    "role": role,
                    "analytical_type": profile.get(
                        "type", str(self.df[name].dtype)
                    ),
                    "pandas_dtype": str(self.df[name].dtype),
                    "unique_count": int(self.df[name].nunique(dropna=True)),
                    "nullable": bool(self.df[name].isna().any()),
                    "sample_values": [
                        str(value)
                        for value in self.df[name].dropna().head(3).tolist()
                    ],
                }
            )

        return {
            "tables": [
                {
                    "name": "Current Dataset",
                    "row_count": int(len(self.df)),
                    "column_count": int(len(self.df.columns)),
                    "fields": fields,
                }
            ],
            "relationships": [],
            "domain": self.semantic_summary.get(
                "detected_domain", "General Business Analytics"
            ),
        }

    @staticmethod
    def validate_field(model: Dict[str, Any], field_name: str) -> bool:
        """Return True only when a field exists in the current model."""
        return any(
            field.get("name") == field_name
            for table in model.get("tables", [])
            for field in table.get("fields", [])
        )
