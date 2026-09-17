import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class EvidenceInsightEngine:
    """
    Evidence-Based Business Insights Engine for YOO PROJECT.
    Follows a strictly grounded architecture:
      Raw Data -> Deterministic Statistical Computations -> Evidence Store -> Natural Language Findings.
    Never fabricates metrics; all insights link to verifiable calculations.
    """

    def __init__(self, df: pd.DataFrame, semantic_summary: Dict[str, Any], profiler_output: Dict[str, Any]):
        self.df = df
        self.semantic = semantic_summary["semantic_roles"]
        self.domain = semantic_summary["detected_domain"]
        self.profiler_summary = profiler_output["summary"]
        self.column_profiles = {cp["name"]: cp for cp in profiler_output["columns"]}
        self.primary_measure = self.semantic.get("primary_measure")
        self.primary_dim = self.semantic.get("primary_dimension")
        self.primary_time = self.semantic.get("primary_time")

    def generate_insights(self) -> List[Dict[str, Any]]:
        """Computes all evidence-backed business insights across 7 analytical dimensions."""
        insights = []

        if not self.primary_measure or self.primary_measure not in self.df.columns:
            return insights

        # 1. RANKING INSIGHT (Top Performer)
        ranking_insight = self._calc_top_performer()
        if ranking_insight:
            insights.append(ranking_insight)

        # 2. CONTRIBUTION / PARETO 80-20 INSIGHT
        pareto_insight = self._calc_pareto_contribution()
        if pareto_insight:
            insights.append(pareto_insight)

        # 3. TEMPORAL TREND INSIGHT
        trend_insight = self._calc_temporal_trend()
        if trend_insight:
            insights.append(trend_insight)

        # 4. ANOMALY / VOLATILITY INSIGHT
        anomaly_insight = self._calc_temporal_anomaly()
        if anomaly_insight:
            insights.append(anomaly_insight)

        # 5. OUTLIERS INSIGHT
        outlier_insight = self._calc_outliers()
        if outlier_insight:
            insights.append(outlier_insight)

        # 6. CORRELATION INSIGHT
        corr_insight = self._calc_correlation()
        if corr_insight:
            insights.append(corr_insight)

        # 7. DATA QUALITY & INTEGRITY INSIGHT
        quality_insight = self._calc_data_quality()
        if quality_insight:
            insights.append(quality_insight)

        return insights

    def _calc_top_performer(self) -> Optional[Dict[str, Any]]:
        """Calculates highest contributing dimension member."""
        if not self.primary_dim or self.primary_dim not in self.df.columns:
            return None

        try:
            grouped = self.df.groupby(self.primary_dim)[self.primary_measure].sum()
            total_sum = grouped.sum()
            if total_sum <= 0:
                return None

            top_entity = grouped.idxmax()
            top_val = float(grouped.max())
            share_pct = round((top_val / total_sum) * 100, 1)

            # Check if there's a runner up
            sorted_g = grouped.sort_values(ascending=False)
            runner_up_text = ""
            if len(sorted_g) > 1:
                runner_up = sorted_g.index[1]
                runner_val = float(sorted_g.iloc[1])
                diff_pct = round(((top_val - runner_val) / runner_val) * 100, 1)
                runner_up_text = f", outpacing #{runner_up} by {diff_pct:+0.1f}%"

            return {
                "id": "insight_ranking",
                "category": "Rankings",
                "type": "Top Performer",
                "title": f"Top Performer: {top_entity}",
                "description": (
                    f"'{top_entity}' generated the highest total {self.primary_measure.replace('_', ' ')} "
                    f"at {top_val:,.2f} ({share_pct}% of total {self.primary_measure.replace('_', ' ')}){runner_up_text}."
                ),
                "metric_label": "Share of Total",
                "metric_value": f"{share_pct}%",
                "badge_color": "#10b981",
                "icon": "award",
                "evidence": {
                    "dimension": self.primary_dim,
                    "top_member": str(top_entity),
                    "value": round(top_val, 2),
                    "total": round(total_sum, 2),
                    "share_percentage": share_pct
                }
            }
        except Exception:
            return None

    def _calc_pareto_contribution(self) -> Optional[Dict[str, Any]]:
        """Computes concentration (e.g. Top 2 or 3 accounting for majority of measure)."""
        if not self.primary_dim or self.primary_dim not in self.df.columns:
            return None

        try:
            grouped = self.df.groupby(self.primary_dim)[self.primary_measure].sum().sort_values(ascending=False)
            total = grouped.sum()
            n_entities = len(grouped)

            if n_entities < 3 or total <= 0:
                return None

            top_n = max(1, min(3, n_entities // 2))
            top_n_sum = grouped.head(top_n).sum()
            concentration_pct = round((top_n_sum / total) * 100, 1)
            top_names = list(grouped.head(top_n).index)

            names_str = ", ".join(f"'{name}'" for name in top_names)
            return {
                "id": "insight_pareto",
                "category": "Contribution",
                "type": "Concentration",
                "title": f"Volume Concentration ({top_n} of {n_entities} {self.primary_dim.replace('_', ' ')}s)",
                "description": (
                    f"The top {top_n} {self.primary_dim.replace('_', ' ')}s ({names_str}) account for "
                    f"{concentration_pct}% of total {self.primary_measure.replace('_', ' ')}. "
                    f"This indicates significant volume concentration."
                ),
                "metric_label": f"Top {top_n} Dominance",
                "metric_value": f"{concentration_pct}%",
                "badge_color": "#6366f1",
                "icon": "pie-chart",
                "evidence": {
                    "top_count": top_n,
                    "total_entities": n_entities,
                    "top_members": [str(x) for x in top_names],
                    "concentration_pct": concentration_pct
                }
            }
        except Exception:
            return None

    def _calc_temporal_trend(self) -> Optional[Dict[str, Any]]:
        """Computes growth, trajectory, and period-over-period direction."""
        if not self.primary_time or self.primary_time not in self.df.columns:
            return None

        try:
            temp = self.df[[self.primary_time, self.primary_measure]].dropna().copy()
            temp["_dt"] = pd.to_datetime(temp[self.primary_time], errors='coerce')
            temp = temp.dropna(subset=["_dt"]).sort_values("_dt")

            if len(temp) < 4:
                return None

            mid = len(temp) // 2
            first_half = temp.iloc[:mid][self.primary_measure].sum()
            second_half = temp.iloc[mid:][self.primary_measure].sum()

            if first_half <= 0:
                return None

            growth_pct = round(((second_half - first_half) / first_half) * 100, 1)
            direction = "increased" if growth_pct >= 0 else "decreased"
            status_color = "#10b981" if growth_pct >= 0 else "#ef4444"

            return {
                "id": "insight_trend",
                "category": "Trends",
                "type": "Temporal Trajectory",
                "title": f"{self.primary_measure.replace('_', ' ').title()} {direction.title()} by {abs(growth_pct)}%",
                "description": (
                    f"Comparing the first half of the timeline to the second half, overall {self.primary_measure.replace('_', ' ')} "
                    f"{direction} from {first_half:,.2f} to {second_half:,.2f} ({growth_pct:+0.1f}% change)."
                ),
                "metric_label": "Trajectory Delta",
                "metric_value": f"{growth_pct:+0.1f}%",
                "badge_color": status_color,
                "icon": "trending-up" if growth_pct >= 0 else "trending-down",
                "evidence": {
                    "first_period_sum": round(first_half, 2),
                    "second_period_sum": round(second_half, 2),
                    "delta_pct": growth_pct
                }
            }
        except Exception:
            return None

    def _calc_temporal_anomaly(self) -> Optional[Dict[str, Any]]:
        """Detects sudden peaks or dips in periodic aggregation."""
        if not self.primary_time or self.primary_time not in self.df.columns:
            return None

        try:
            temp = self.df[[self.primary_time, self.primary_measure]].dropna().copy()
            temp["_dt"] = pd.to_datetime(temp[self.primary_time], errors='coerce')
            temp = temp.dropna(subset=["_dt"])
            temp["_month"] = temp["_dt"].dt.to_period("M").astype(str)

            monthly = temp.groupby("_month")[self.primary_measure].sum()
            if len(monthly) < 3:
                return None

            mean_val = monthly.mean()
            std_val = monthly.std()
            if std_val == 0:
                return None

            # Look for month with maximum absolute deviation
            deviations = (monthly - mean_val) / std_val
            extreme_month = deviations.abs().idxmax()
            extreme_dev = deviations[extreme_month]
            extreme_val = monthly[extreme_month]

            if abs(extreme_dev) >= 1.2:
                flavor = "peak" if extreme_dev > 0 else "trough"
                dev_pct = round(((extreme_val - mean_val) / mean_val) * 100, 1)

                return {
                    "id": "insight_anomaly",
                    "category": "Anomalies",
                    "type": "Periodic Anomaly",
                    "title": f"Significant {flavor.title()} in {extreme_month}",
                    "description": (
                        f"A prominent {flavor} was detected in {extreme_month} with {extreme_val:,.2f} "
                        f"({dev_pct:+0.1f}% relative to the monthly average of {mean_val:,.2f})."
                    ),
                    "metric_label": f"Deviation from Avg",
                    "metric_value": f"{dev_pct:+0.1f}%",
                    "badge_color": "#f59e0b",
                    "icon": "alert-circle",
                    "evidence": {
                        "period": extreme_month,
                        "value": round(extreme_val, 2),
                        "monthly_average": round(mean_val, 2),
                        "z_score": round(extreme_dev, 2)
                    }
                }
        except Exception:
            pass
        return None

    def _calc_outliers(self) -> Optional[Dict[str, Any]]:
        """Reports outlier occurrences in the primary measure."""
        profile = self.column_profiles.get(self.primary_measure, {})
        outlier_count = profile.get("outlier_count", 0)
        outlier_pct = profile.get("outlier_percentage", 0.0)

        if outlier_count > 0:
            samples = profile.get("sample_outliers", [])
            max_val = profile.get("max", 0)
            median_val = profile.get("median", 0)

            return {
                "id": "insight_outliers",
                "category": "Outliers",
                "type": "Statistical Outliers",
                "title": f"{outlier_count} Statistical Outliers Identified",
                "description": (
                    f"{outlier_count} records ({outlier_pct}% of dataset) exceed 1.5x the Interquartile Range (IQR). "
                    f"The highest outlier reached {max_val:,.2f} versus a median of {median_val:,.2f}."
                ),
                "metric_label": "Outliers Detected",
                "metric_value": str(outlier_count),
                "badge_color": "#ec4899",
                "icon": "zap",
                "evidence": {
                    "outlier_count": outlier_count,
                    "outlier_percentage": outlier_pct,
                    "max_value": max_val,
                    "median_value": median_val,
                    "sample_values": samples
                }
            }
        return None

    def _calc_correlation(self) -> Optional[Dict[str, Any]]:
        """Computes strongest numerical correlation."""
        sec_measures = self.semantic.get("secondary_measures", [])
        if not sec_measures:
            return None

        best_m = None
        best_r = 0.0

        for m in sec_measures:
            try:
                sub = self.df[[self.primary_measure, m]].dropna()
                if len(sub) >= 6:
                    r = np.corrcoef(sub[self.primary_measure], sub[m])[0, 1]
                    if not np.isnan(r) and abs(r) > abs(best_r):
                        best_r = r
                        best_m = m
            except Exception:
                pass

        if best_m and abs(best_r) >= 0.35:
            strength = "strong" if abs(best_r) >= 0.7 else "moderate"
            direction = "positive" if best_r > 0 else "inverse (negative)"

            return {
                "id": "insight_correlation",
                "category": "Correlations",
                "type": "Metric Correlation",
                "title": f"{strength.title()} {direction.title()} Correlation ({best_r:0.2f})",
                "description": (
                    f"A {strength} {direction} linear relationship was calculated between "
                    f"'{self.primary_measure.replace('_', ' ')}' and '{best_m.replace('_', ' ')}' (Pearson r = {best_r:0.2f})."
                ),
                "metric_label": "Correlation (r)",
                "metric_value": f"{best_r:0.2f}",
                "badge_color": "#38bdf8",
                "icon": "link",
                "evidence": {
                    "measure_1": self.primary_measure,
                    "measure_2": best_m,
                    "pearson_r": round(best_r, 3),
                    "strength": strength
                }
            }
        return None

    def _calc_data_quality(self) -> Optional[Dict[str, Any]]:
        """Data Quality audit insight."""
        score = self.profiler_summary.get("data_health_score", 100)
        null_pct = self.profiler_summary.get("overall_null_percentage", 0.0)
        dup_cnt = self.profiler_summary.get("duplicate_rows", 0)

        return {
            "id": "insight_quality",
            "category": "Data Quality",
            "type": "Integrity Check",
            "title": f"Data Health Score: {score}% ({self.profiler_summary.get('data_health_status')})",
            "description": (
                f"Dataset exhibits {100 - null_pct:0.1f}% completeness with {null_pct}% missing values "
                f"and {dup_cnt} duplicate rows across {self.profiler_summary.get('total_rows')} records."
            ),
            "metric_label": "Health Score",
            "metric_value": f"{score}%",
            "badge_color": self.profiler_summary.get("data_health_color", "#10b981"),
            "icon": "shield-check",
            "evidence": {
                "health_score": score,
                "overall_null_pct": null_pct,
                "duplicates": dup_cnt,
                "total_rows": self.profiler_summary.get("total_rows")
            }
        }
