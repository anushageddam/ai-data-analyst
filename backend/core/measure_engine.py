from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

import pandas as pd


class MeasureError(ValueError):
    """Raised when a measure cannot be validated or evaluated."""


class MeasureEngine:
    """Small, deterministic measure engine inspired by common BI aggregations."""

    FUNCTIONS: Dict[str, Callable[..., float]] = {}

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _require_column(self, column: str) -> pd.Series:
        if not isinstance(column, str) or not column.strip():
            raise MeasureError("A field name is required.")
        if column not in self.df.columns:
            raise MeasureError(f"Field '{column}' does not exist in the current dataset.")
        return self.df[column]

    def _numeric(self, column: str) -> pd.Series:
        series = self._require_column(column)
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            raise MeasureError(f"Field '{column}' must contain numeric values for this calculation.")
        return numeric.dropna()

    def evaluate(self, function: str, column: Optional[str] = None) -> float:
        fn = str(function).strip().upper()
        if fn == "COUNT":
            return float(self._require_column(column).notna().sum())
        if fn == "DISTINCTCOUNT":
            return float(self._require_column(column).dropna().nunique())
        values = self._numeric(column)
        operations = {
            "SUM": values.sum,
            "AVERAGE": values.mean,
            "MIN": values.min,
            "MAX": values.max,
            "MEDIAN": values.median,
        }
        if fn not in operations:
            raise MeasureError(
                f"Unsupported function '{function}'. Supported functions: "
                + ", ".join(sorted(list(operations) + ["COUNT", "DISTINCTCOUNT", "DIVIDE", "IF"]))
                + "."
            )
        return float(operations[fn]())

    def divide(self, numerator: float, denominator: float, alternate_result: float = 0.0) -> float:
        if denominator == 0:
            return float(alternate_result)
        return float(numerator) / float(denominator)

    def ratio(self, numerator_column: str, denominator_column: str, as_percentage: bool = False) -> float:
        numerator = self.evaluate("SUM", numerator_column)
        denominator = self.evaluate("SUM", denominator_column)
        result = self.divide(numerator, denominator)
        return result * 100.0 if as_percentage else result

    def filtered(self, function: str, column: str, filters: Dict[str, Any]) -> float:
        if not isinstance(filters, dict):
            raise MeasureError("Filters must be supplied as a field-to-value dictionary.")

        mask = pd.Series(True, index=self.df.index)
        for field, expected in filters.items():
            series = self._require_column(field)
            if isinstance(expected, (list, tuple, set)):
                mask &= series.isin(list(expected))
            else:
                mask &= series.eq(expected)

        subset = self.df.loc[mask]
        return MeasureEngine(subset).evaluate(function, column)

    @staticmethod
    def validate_definition(function: str, column: Optional[str], df: pd.DataFrame) -> Dict[str, Any]:
        fn = str(function).strip().upper()
        supported = {"SUM", "AVERAGE", "MIN", "MAX", "COUNT", "DISTINCTCOUNT", "MEDIAN", "DIVIDE", "IF"}
        errors = []

        if fn not in supported:
            errors.append(f"Unsupported function '{function}'.")
        if fn not in {"DIVIDE", "IF"} and (not column or column not in df.columns):
            errors.append(f"Field '{column}' does not exist in the current dataset.")

        if column in df.columns and fn in {"SUM", "AVERAGE", "MIN", "MAX", "MEDIAN"}:
            if pd.to_numeric(df[column], errors="coerce").notna().sum() == 0:
                errors.append(f"Field '{column}' is not numeric.")

        return {"valid": not errors, "errors": errors}

    @staticmethod
    def parse_simple_definition(definition: str) -> Dict[str, str]:
        """Parse simple reusable definitions such as SUM(Net_Value)."""
        pattern = r"^\s*([A-Za-z]+)\s*\(\s*['\"]?(.+?)['\"]?\s*\)\s*$"
        match = re.match(pattern, definition or "")
        if not match:
            raise MeasureError("Use the format FUNCTION(Field), for example SUM(Net_Value).")
        return {"function": match.group(1).upper(), "column": match.group(2).strip()}


class CustomMeasureEngine:
    """Deterministic reusable custom-measure definitions for simple BI formulas."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.measures: Dict[str, Dict[str, Any]] = {}

    def create(self, name: str, formula: str) -> Dict[str, Any]:
        clean_name = str(name or "").strip()
        clean_formula = str(formula or "").strip()
        if not clean_name:
            raise MeasureError("A custom measure name is required.")
        if not clean_formula:
            raise MeasureError("A custom measure formula is required.")

        tokens = [t.strip() for t in re.split(r"([+*/-])", clean_formula) if t.strip()]
        fields = [t for t in tokens if t not in {"+", "-", "*", "/"}]
        if not fields:
            raise MeasureError("Formula must contain at least one field.")
        for field in fields:
            if field not in self.df.columns:
                raise MeasureError(f"Field '{field}' does not exist in the current dataset.")
            if pd.to_numeric(self.df[field], errors="coerce").notna().sum() == 0:
                raise MeasureError(f"Field '{field}' must be numeric for a custom measure.")

        values = {field: float(pd.to_numeric(self.df[field], errors="coerce").sum()) for field in fields}
        expression = clean_formula
        for field in sorted(fields, key=len, reverse=True):
            expression = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])",
                str(values[field]),
                expression,
            )
        try:
            result = float(self._safe_eval(expression))
        except ZeroDivisionError:
            raise MeasureError("Custom measure contains division by zero.")
        except Exception as exc:
            raise MeasureError(f"Invalid custom measure formula: {exc}")

        definition = {"name": clean_name, "formula": clean_formula, "value": result}
        self.measures[clean_name] = definition
        return definition

    @staticmethod
    def _safe_eval(expression: str) -> float:
        allowed = set("0123456789eE.+-*/() ")
        if not expression or any(char not in allowed for char in expression):
            raise MeasureError("Only numeric arithmetic (+, -, *, /) is supported.")
        return eval(expression, {"__builtins__": {}}, {})
