import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class VisDecisionEngine:
    """
    Intelligent Visualization Decision Engine for YOO PROJECT.
    Selects optimal, non-redundant Power BI-style charts based on cardinality,
    data types, temporal availability, ranking requirements, and analytical relationships.
    Outputs declarative Plotly specifications.
    """

    COLOR_PALETTE = [
        "#6366f1", "#38bdf8", "#10b981", "#f59e0b", "#ec4899",
        "#8b5cf6", "#14b8a6", "#f97316", "#06b6d4", "#a855f7"
    ]

    DARK_THEME_LAYOUT = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "color": "#94a3b8", "size": 12},
        "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.06)", "zerolinecolor": "rgba(255,255,255,0.1)"},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.06)", "zerolinecolor": "rgba(255,255,255,0.1)"},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#cbd5e1"}},
        "hoverlabel": {"bgcolor": "#1e293b", "font": {"color": "#f8fafc", "family": "Inter"}}
    }

    def __init__(self, df: pd.DataFrame, semantic_summary: Dict[str, Any], profiler_output: Dict[str, Any]):
        self.df = df
        self.semantic = semantic_summary["semantic_roles"]
        self.column_profiles = {cp["name"]: cp for cp in profiler_output["columns"]}
        self.domain = semantic_summary["detected_domain"]

    def generate_dashboard_visuals(self) -> List[Dict[str, Any]]:
        """Produces 4-6 intelligent, diverse, non-redundant visualizations for the main dashboard."""
        visuals = []
        measures = self.semantic.get("all_measures", [])
        primary_measure = self.semantic.get("primary_measure")
        secondary_measures = self.semantic.get("secondary_measures", [])
        time_dim = self.semantic.get("primary_time")
        primary_dim = self.semantic.get("primary_dimension")
        all_dims = self.semantic.get("all_dimensions", [])

        if not primary_measure:
            return visuals

        # ---------------------------------------------------------------------
        # 1. PRIMARY TIME-SERIES TREND (Line / Area Chart)
        # ---------------------------------------------------------------------
        if time_dim and time_dim in self.df.columns:
            trend_vis = self._build_time_series_visual(time_dim, primary_measure, secondary_measures)
            if trend_vis:
                visuals.append(trend_vis)

        # ---------------------------------------------------------------------
        # 2. PRIMARY CATEGORICAL BREAKDOWN (Donut vs Bar vs Top-N Horizontal)
        # ---------------------------------------------------------------------
        if primary_dim and primary_dim in self.df.columns:
            cat_vis = self._build_categorical_visual(primary_dim, primary_measure)
            if cat_vis:
                visuals.append(cat_vis)

        # ---------------------------------------------------------------------
        # 3. SECONDARY DIMENSION / COMPARISON BREAKDOWN
        # ---------------------------------------------------------------------
        other_dims = [d for d in all_dims if d != primary_dim]
        if other_dims:
            sec_dim = other_dims[0]
            # If 2 measures available, create Grouped Bar Chart; else standard breakdown
            if secondary_measures and len(self.df[sec_dim].dropna().unique()) <= 12:
                sec_vis = self._build_grouped_bar_visual(sec_dim, primary_measure, secondary_measures[0])
            else:
                sec_vis = self._build_categorical_visual(sec_dim, primary_measure)
            if sec_vis:
                visuals.append(sec_vis)

        # ---------------------------------------------------------------------
        # 4. CORRELATION / SCATTER PLOT (When 2 Continuous Measures Exist)
        # ---------------------------------------------------------------------
        if secondary_measures:
            sec_measure = secondary_measures[0]
            scatter_vis = self._build_scatter_visual(primary_measure, sec_measure, primary_dim)
            if scatter_vis:
                visuals.append(scatter_vis)

        # ---------------------------------------------------------------------
        # 5. THIRD DIMENSION / REGIONAL / DEPT BREAKDOWN
        # ---------------------------------------------------------------------
        if len(other_dims) > 1 and len(visuals) < 5:
            third_dim = other_dims[1]
            third_vis = self._build_categorical_visual(third_dim, primary_measure)
            if third_vis:
                visuals.append(third_vis)

        # ---------------------------------------------------------------------
        # 6. CORRELATION MATRIX / HEATMAP (When 3+ Measures Exist)
        # ---------------------------------------------------------------------
        if len(measures) >= 3 and len(visuals) < 6:
            heatmap_vis = self._build_correlation_heatmap(measures[:6])
            if heatmap_vis:
                visuals.append(heatmap_vis)

        return visuals

    def _build_time_series_visual(self, time_col: str, measure_col: str, sec_measures: List[str]) -> Optional[Dict[str, Any]]:
        """Generates an aggregated line / area chart with temporal intelligence."""
        try:
            temp_df = self.df[[time_col, measure_col]].dropna().copy()
            temp_df["_dt"] = pd.to_datetime(temp_df[time_col], errors='coerce')
            temp_df = temp_df.dropna(subset=["_dt"]).sort_values("_dt")

            if len(temp_df) < 3:
                return None

            # Determine best aggregation frequency (Month / Day / Year)
            date_range_days = (temp_df["_dt"].max() - temp_df["_dt"].min()).days
            if date_range_days > 730:
                temp_df["_period"] = temp_df["_dt"].dt.to_period("Q").astype(str)
                freq_label = "Quarterly"
            elif date_range_days > 45:
                temp_df["_period"] = temp_df["_dt"].dt.to_period("M").astype(str)
                freq_label = "Monthly"
            else:
                temp_df["_period"] = temp_df["_dt"].dt.strftime('%Y-%m-%d')
                freq_label = "Daily"

            grouped = temp_df.groupby("_period")[measure_col].sum().reset_index()

            x_vals = grouped["_period"].tolist()
            y_vals = [round(float(v), 2) for v in grouped[measure_col].tolist()]

            traces = [{
                "x": x_vals,
                "y": y_vals,
                "type": "scatter",
                "mode": "lines+markers",
                "name": measure_col.replace('_', ' ').title(),
                "line": {"color": "#6366f1", "width": 3, "shape": "spline"},
                "fill": "tozeroy",
                "fillcolor": "rgba(99, 102, 241, 0.12)",
                "marker": {"size": 6, "color": "#818cf8"}
            }]

            layout = {
                **self.DARK_THEME_LAYOUT,
                "title": f"{freq_label} {measure_col.replace('_', ' ').title()} Trend",
                "xaxis": {**self.DARK_THEME_LAYOUT["xaxis"], "title": "Time Period"},
                "yaxis": {**self.DARK_THEME_LAYOUT["yaxis"], "title": measure_col.replace('_', ' ').title()}
            }

            return {
                "id": "vis_primary_trend",
                "title": f"{freq_label} {measure_col.replace('_', ' ').title()} Trend",
                "chart_type": "line_area",
                "section": "Trends",
                "description": f"Historical {freq_label.lower()} trajectory of {measure_col.replace('_', ' ')}.",
                "plotly_spec": {"data": traces, "layout": layout}
            }
        except Exception:
            return None

    def _build_categorical_visual(self, dim_col: str, measure_col: str) -> Optional[Dict[str, Any]]:
        """
        Decision rule:
        - <= 5 categories -> Donut Chart
        - 6 to 15 categories -> Vertical Bar Chart
        - > 15 categories -> Top-10 Horizontal Bar Chart
        """
        try:
            grouped = self.df.groupby(dim_col)[measure_col].sum().reset_index()
            grouped = grouped.sort_values(measure_col, ascending=False).reset_index(drop=True)
            n_cats = len(grouped)

            if n_cats == 0:
                return None

            # CASE A: <= 5 Categories -> Donut Chart
            if n_cats <= 5:
                labels = [str(l) for l in grouped[dim_col].tolist()]
                values = [round(float(v), 2) for v in grouped[measure_col].tolist()]
                traces = [{
                    "labels": labels,
                    "values": values,
                    "type": "pie",
                    "hole": 0.58,
                    "marker": {"colors": self.COLOR_PALETTE[:n_cats]},
                    "textinfo": "label+percent",
                    "textfont": {"color": "#ffffff", "family": "Inter"},
                    "hoverinfo": "label+value+percent"
                }]
                layout = {
                    **self.DARK_THEME_LAYOUT,
                    "title": f"{measure_col.replace('_', ' ').title()} Contribution by {dim_col.replace('_', ' ').title()}",
                    "showlegend": True
                }
                return {
                    "id": f"vis_donut_{dim_col}",
                    "title": f"{measure_col.replace('_', ' ').title()} Share by {dim_col.replace('_', ' ').title()}",
                    "chart_type": "donut",
                    "section": "Breakdown",
                    "description": f"Part-to-whole contribution across {dim_col.replace('_', ' ')}.",
                    "plotly_spec": {"data": traces, "layout": layout}
                }

            # CASE B: 6 to 15 Categories -> Vertical Column / Bar Chart
            elif 6 <= n_cats <= 15:
                x_vals = [str(l) for l in grouped[dim_col].tolist()]
                y_vals = [round(float(v), 2) for v in grouped[measure_col].tolist()]
                traces = [{
                    "x": x_vals,
                    "y": y_vals,
                    "type": "bar",
                    "marker": {
                        "color": "#38bdf8",
                        "line": {"color": "#0284c7", "width": 1}
                    }
                }]
                layout = {
                    **self.DARK_THEME_LAYOUT,
                    "title": f"{measure_col.replace('_', ' ').title()} by {dim_col.replace('_', ' ').title()}",
                    "xaxis": {**self.DARK_THEME_LAYOUT["xaxis"], "title": dim_col.replace('_', ' ').title()},
                    "yaxis": {**self.DARK_THEME_LAYOUT["yaxis"], "title": measure_col.replace('_', ' ').title()}
                }
                return {
                    "id": f"vis_bar_{dim_col}",
                    "title": f"{measure_col.replace('_', ' ').title()} by {dim_col.replace('_', ' ').title()}",
                    "chart_type": "bar",
                    "section": "Performance",
                    "description": f"Comparison of {measure_col.replace('_', ' ')} across {dim_col.replace('_', ' ')}.",
                    "plotly_spec": {"data": traces, "layout": layout}
                }

            # CASE C: > 15 Categories -> Top 10 Horizontal Bar Chart (Prevents crowded pie charts)
            else:
                top_10 = grouped.head(10).iloc[::-1]  # Reverse for bottom-to-top ranking
                y_vals = [str(l) for l in top_10[dim_col].tolist()]
                x_vals = [round(float(v), 2) for v in top_10[measure_col].tolist()]
                traces = [{
                    "x": x_vals,
                    "y": y_vals,
                    "type": "bar",
                    "orientation": "h",
                    "marker": {
                        "color": "#10b981",
                        "line": {"color": "#059669", "width": 1}
                    }
                }]
                layout = {
                    **self.DARK_THEME_LAYOUT,
                    "title": f"Top 10 {dim_col.replace('_', ' ').title()} by {measure_col.replace('_', ' ').title()}",
                    "xaxis": {**self.DARK_THEME_LAYOUT["xaxis"], "title": measure_col.replace('_', ' ').title()},
                    "yaxis": {**self.DARK_THEME_LAYOUT["yaxis"], "title": "", "automargin": True}
                }
                return {
                    "id": f"vis_top10_{dim_col}",
                    "title": f"Top 10 {dim_col.replace('_', ' ').title()}",
                    "chart_type": "horizontal_bar",
                    "section": "Performance",
                    "description": f"Top 10 highest-ranked {dim_col.replace('_', ' ')} by {measure_col.replace('_', ' ')}.",
                    "plotly_spec": {"data": traces, "layout": layout}
                }
        except Exception:
            return None

    def _build_grouped_bar_visual(self, dim_col: str, m1: str, m2: str) -> Optional[Dict[str, Any]]:
        """Generates a Grouped Bar Chart comparing two measures across categories."""
        try:
            grouped = self.df.groupby(dim_col)[[m1, m2]].sum().reset_index().head(10)
            cats = [str(c) for c in grouped[dim_col].tolist()]

            traces = [
                {
                    "x": cats,
                    "y": [round(float(v), 2) for v in grouped[m1].tolist()],
                    "name": m1.replace('_', ' ').title(),
                    "type": "bar",
                    "marker": {"color": "#6366f1"}
                },
                {
                    "x": cats,
                    "y": [round(float(v), 2) for v in grouped[m2].tolist()],
                    "name": m2.replace('_', ' ').title(),
                    "type": "bar",
                    "marker": {"color": "#38bdf8"}
                }
            ]

            layout = {
                **self.DARK_THEME_LAYOUT,
                "title": f"{m1.replace('_', ' ').title()} & {m2.replace('_', ' ').title()} by {dim_col.replace('_', ' ').title()}",
                "barmode": "group",
                "xaxis": {**self.DARK_THEME_LAYOUT["xaxis"], "title": dim_col.replace('_', ' ').title()},
                "yaxis": {**self.DARK_THEME_LAYOUT["yaxis"], "title": "Values"}
            }

            return {
                "id": f"vis_grouped_{dim_col}",
                "title": f"{m1.replace('_', ' ').title()} vs {m2.replace('_', ' ').title()} by {dim_col.replace('_', ' ').title()}",
                "chart_type": "grouped_bar",
                "section": "Analysis",
                "description": f"Side-by-side comparative analysis across {dim_col.replace('_', ' ')}.",
                "plotly_spec": {"data": traces, "layout": layout}
            }
        except Exception:
            return None

    def _build_scatter_visual(self, m1: str, m2: str, hover_dim: Optional[str]) -> Optional[Dict[str, Any]]:
        """Generates a Scatter Plot with linear trendline for correlation analysis."""
        try:
            cols = [m1, m2] + ([hover_dim] if hover_dim and hover_dim in self.df.columns else [])
            sub = self.df[cols].dropna().copy()
            if len(sub) < 5:
                return None

            x = sub[m1].astype(float).values
            y = sub[m2].astype(float).values
            hover_text = sub[hover_dim].astype(str).values if hover_dim and hover_dim in sub else None

            # Calculate correlation coefficient
            corr = np.corrcoef(x, y)[0, 1] if len(x) > 2 else 0.0

            # Linear regression trendline
            slope, intercept = np.polyfit(x, y, 1)
            x_trend = np.linspace(x.min(), x.max(), 50)
            y_trend = slope * x_trend + intercept

            traces = [
                {
                    "x": [round(float(v), 2) for v in x],
                    "y": [round(float(v), 2) for v in y],
                    "mode": "markers",
                    "type": "scatter",
                    "name": "Data Points",
                    "text": hover_text.tolist() if hover_text is not None else None,
                    "marker": {"size": 8, "color": "#f59e0b", "opacity": 0.8, "line": {"color": "#b45309", "width": 1}}
                },
                {
                    "x": [round(float(v), 2) for v in x_trend],
                    "y": [round(float(v), 2) for v in y_trend],
                    "mode": "lines",
                    "type": "scatter",
                    "name": f"Trendline (r = {corr:.2f})",
                    "line": {"color": "#ec4899", "width": 2, "dash": "dash"}
                }
            ]

            layout = {
                **self.DARK_THEME_LAYOUT,
                "title": f"Correlation: {m1.replace('_', ' ').title()} vs {m2.replace('_', ' ').title()}",
                "xaxis": {**self.DARK_THEME_LAYOUT["xaxis"], "title": m1.replace('_', ' ').title()},
                "yaxis": {**self.DARK_THEME_LAYOUT["yaxis"], "title": m2.replace('_', ' ').title()}
            }

            return {
                "id": "vis_scatter_corr",
                "title": f"{m1.replace('_', ' ').title()} vs {m2.replace('_', ' ').title()}",
                "chart_type": "scatter",
                "section": "Analysis",
                "description": f"Correlation analysis (Pearson r = {corr:.2f}).",
                "plotly_spec": {"data": traces, "layout": layout}
            }
        except Exception:
            return None

    def _build_correlation_heatmap(self, measures: List[str]) -> Optional[Dict[str, Any]]:
        """Generates a Correlation Matrix Heatmap."""
        try:
            num_df = self.df[measures].apply(pd.to_numeric, errors='coerce').dropna()
            if len(num_df) < 5:
                return None

            corr_matrix = num_df.corr().round(2)
            z_vals = corr_matrix.values.tolist()
            labels = [m.replace('_', ' ').title() for m in measures]

            traces = [{
                "z": z_vals,
                "x": labels,
                "y": labels,
                "type": "heatmap",
                "colorscale": [[0, "#312e81"], [0.5, "#1e293b"], [1, "#06b6d4"]],
                "showscale": True
            }]

            layout = {
                **self.DARK_THEME_LAYOUT,
                "title": "Numerical Correlation Heatmap",
                "xaxis": {**self.DARK_THEME_LAYOUT["xaxis"], "title": ""},
                "yaxis": {**self.DARK_THEME_LAYOUT["yaxis"], "title": ""}
            }

            return {
                "id": "vis_heatmap",
                "title": "Metric Correlation Matrix",
                "chart_type": "heatmap",
                "section": "Analysis",
                "description": "Multi-variable correlation strength between all numeric measures.",
                "plotly_spec": {"data": traces, "layout": layout}
            }
        except Exception:
            return None
