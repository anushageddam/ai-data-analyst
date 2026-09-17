import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class DataProfiler:
    """
    Automatic Data Profiler for YOO PROJECT.
    Inspects columns, determines exact semantic and storage data types,
    detects data quality issues, outliers, and computes an overall Data Health Score.
    """

    ID_PATTERNS = [
        r'.*id$', r'^id_.*', r'.*_no$', r'.*_num$', r'.*_number$',
        r'^po_.*', r'^order_.*', r'^invoice_.*', r'^trans_.*', r'^customer_id$',
        r'^sku$', r'^code$', r'.*_code$', r'^guid$', r'^uuid$'
    ]

    DATE_PATTERNS = [
        r'.*date.*', r'.*time.*', r'.*timestamp.*', r'^year$', r'^month$',
        r'^quarter$', r'^period$', r'.*_dt$', r'.*_at$'
    ]

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = df.copy()
        self._standardize_dates()

    def _standardize_dates(self):
        """Attempts to convert detected date columns into pd.to_datetime."""
        for col in self.df.columns:
            col_lower = str(col).lower()
            if any(re.match(pat, col_lower) for pat in self.DATE_PATTERNS):
                try:
                    converted = pd.to_datetime(self.df[col], errors='coerce')
                    if converted.notna().mean() >= 0.65:
                        self.df[col] = converted
                except Exception:
                    pass

    def detect_column_type(self, col: str) -> str:
        """
        Determines the analytical type of a column:
        - Numerical
        - Categorical
        - DateTime
        - Boolean
        - ID
        - Text
        """
        series = self.df[col]
        col_lower = str(col).lower()
        non_null = series.dropna()
        n_total = len(series)
        n_unique = series.nunique()

        # 1. ID detection (name pattern or extreme uniqueness in integer/alphanumeric)
        is_named_id = any(re.match(pat, col_lower) for pat in self.ID_PATTERNS)
        if is_named_id:
            return "ID"

        # 2. Boolean detection
        if pd.api.types.is_bool_dtype(series):
            return "Boolean"
        if n_unique == 2 and n_total > 5:
            vals = set(non_null.astype(str).str.lower().unique())
            if vals.issubset({'true', 'false', '0', '1', 'yes', 'no', 'y', 'n'}):
                return "Boolean"

        # 3. DateTime detection
        if pd.api.types.is_datetime64_any_dtype(series):
            return "DateTime"
        
        # Try datetime conversion if named like date
        if any(re.match(pat, col_lower) for pat in self.DATE_PATTERNS):
            try:
                converted = pd.to_datetime(series, errors='coerce')
                if converted.notna().mean() >= 0.70:
                    return "DateTime"
            except Exception:
                pass

        # 4. Numerical
        if pd.api.types.is_numeric_dtype(series):
            # Check if high uniqueness integer looks like an ID
            if pd.api.types.is_integer_dtype(series) and n_total > 10 and (n_unique / n_total) > 0.95:
                # If column name has no measure keywords (sales, cost, qty, price, score)
                if not any(k in col_lower for k in ['amount', 'sales', 'val', 'price', 'cost', 'qty', 'quantity', 'rate', 'discount', 'margin', 'profit', 'spend']):
                    return "ID"
            return "Numerical"

        # 5. High-cardinality text vs categorical
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            if n_total > 10 and (n_unique / n_total) > 0.92:
                # Either unique ID or free-form text
                avg_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
                if avg_len > 30:
                    return "Text"
                return "ID"
            return "Categorical"

        return "Categorical"

    def compute_outliers(self, series: pd.Series) -> Tuple[int, float, List[float]]:
        """Computes outliers using Interquartile Range (IQR) method."""
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) < 8:
            return 0, 0.0, []
        
        q25, q75 = np.percentile(numeric_series, [25, 75])
        iqr = q75 - q25
        if iqr == 0:
            return 0, 0.0, []
            
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(numeric_series)) * 100, 2)
        sample_outliers = [round(float(v), 2) for v in numeric_series[outlier_mask].head(5).tolist()]
        return outlier_count, outlier_pct, sample_outliers

    def profile_column(self, col: str) -> Dict[str, Any]:
        """Deep profiling of a single column."""
        series = self.df[col]
        total_rows = len(series)
        null_count = int(series.isnull().sum())
        null_pct = round((null_count / total_rows * 100) if total_rows > 0 else 0, 2)
        unique_count = int(series.nunique())
        cardinality_ratio = round((unique_count / total_rows) if total_rows > 0 else 0, 4)
        col_type = self.detect_column_type(col)

        # Top 5 most frequent values
        non_null_s = series.dropna()
        top_freq = []
        if len(non_null_s) > 0:
            vc = non_null_s.astype(str).value_counts().head(5)
            for val, cnt in vc.items():
                top_freq.append({
                    "value": str(val)[:30],
                    "count": int(cnt),
                    "percentage": round((cnt / total_rows) * 100, 1)
                })

        info: Dict[str, Any] = {
            "name": col,
            "type": col_type,
            "pandas_dtype": str(series.dtype),
            "total_rows": total_rows,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "cardinality_ratio": cardinality_ratio,
            "is_constant": unique_count == 1,
            "is_empty": null_count == total_rows,
            "top_frequencies": top_freq,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "sample_outliers": []
        }

        if col_type == "Numerical":
            num_s = pd.to_numeric(series, errors='coerce').dropna()
            if len(num_s) > 0:
                info["min"] = round(float(num_s.min()), 2)
                info["max"] = round(float(num_s.max()), 2)
                info["mean"] = round(float(num_s.mean()), 2)
                info["median"] = round(float(num_s.median()), 2)
                info["std"] = round(float(num_s.std()), 2) if len(num_s) > 1 else 0.0
                out_cnt, out_pct, out_samples = self.compute_outliers(num_s)
                info["outlier_count"] = out_cnt
                info["outlier_percentage"] = out_pct
                info["sample_outliers"] = out_samples

        elif col_type == "DateTime":
            dt_s = pd.to_datetime(series, errors='coerce').dropna()
            if len(dt_s) > 0:
                info["min"] = str(dt_s.min().strftime('%Y-%m-%d'))
                info["max"] = str(dt_s.max().strftime('%Y-%m-%d'))

        return info

    def profile_dataset(self) -> Dict[str, Any]:
        """Profiles the entire dataset and computes the Data Health Score."""
        total_rows = int(len(self.df))
        total_cols = int(len(self.df.columns))
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows * 100) if total_rows > 0 else 0, 2)

        column_profiles = [self.profile_column(c) for c in self.df.columns]

        # Calculate dataset-wide health metrics
        total_cells = total_rows * total_cols
        total_nulls = sum(cp["null_count"] for cp in column_profiles)
        overall_null_pct = round((total_nulls / total_cells * 100) if total_cells > 0 else 0, 2)
        total_outliers = sum(cp["outlier_count"] for cp in column_profiles)
        empty_cols = sum(1 for cp in column_profiles if cp["is_empty"])
        constant_cols = sum(1 for cp in column_profiles if cp["is_constant"])

        # Health score algorithm (starts at 100)
        # Deduct for missing data, duplicates, empty columns, extreme outliers
        health_score = 100.0
        health_score -= min(35.0, overall_null_pct * 1.5)
        health_score -= min(25.0, duplicate_pct * 2.5)
        health_score -= (empty_cols * 5.0)
        health_score -= (constant_cols * 3.0)
        if total_rows > 0:
            outlier_ratio = (total_outliers / total_rows) * 100
            health_score -= min(15.0, outlier_ratio * 0.5)

        health_score = max(5.0, min(100.0, round(health_score, 1)))

        if health_score >= 90:
            health_status = "Excellent"
            health_color = "#10b981"
        elif health_score >= 75:
            health_status = "Good"
            health_color = "#3b82f6"
        elif health_score >= 55:
            health_status = "Fair"
            health_color = "#f59e0b"
        else:
            health_status = "Needs Attention"
            health_color = "#ef4444"

        # Group columns by type
        columns_by_type: Dict[str, List[str]] = {
            "Numerical": [],
            "Categorical": [],
            "DateTime": [],
            "Boolean": [],
            "ID": [],
            "Text": []
        }
        for cp in column_profiles:
            columns_by_type[cp["type"]].append(cp["name"])

        return {
            "summary": {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": duplicate_pct,
                "total_null_cells": total_nulls,
                "overall_null_percentage": overall_null_pct,
                "total_outliers_detected": total_outliers,
                "empty_columns_count": empty_cols,
                "constant_columns_count": constant_cols,
                "data_health_score": health_score,
                "data_health_status": health_status,
                "data_health_color": health_color
            },
            "columns": column_profiles,
            "columns_by_type": columns_by_type
        }
