import re
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class ChatContext:
    """Maintains multi-turn conversational session context."""
    def __init__(self):
        self.last_dimension: Optional[str] = None
        self.last_measure: Optional[str] = None
        self.last_intent: Optional[str] = None
        self.last_result_df: Optional[pd.DataFrame] = None
        self.last_chart_spec: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, str]] = []

    def update(self, intent: str, dimension: Optional[str], measure: Optional[str], result_df: Optional[pd.DataFrame] = None, chart: Optional[Dict[str, Any]] = None):
        self.last_intent = intent
        if dimension:
            self.last_dimension = dimension
        if measure:
            self.last_measure = measure
        if result_df is not None:
            self.last_result_df = result_df
        if chart is not None:
            self.last_chart_spec = chart

    def add_turn(self, role: str, message: str):
        self.history.append({"role": role, "content": message})
        if len(self.history) > 10:
            self.history = self.history[-10:]


class AIChatEngine:
    """
    Context-Aware Conversational BI Engine for YOO PROJECT.
    Understands questions about dimensions, measures, trends, rankings, and forecasts.
    Executes real mathematical calculations over the active DataFrame and optionally returns inline charts.
    Strictly prohibits numerical hallucinations.
    """

    def __init__(self, df: pd.DataFrame, semantic_summary: Dict[str, Any], forecast_engine=None):
        self.df = df
        self.semantic = semantic_summary["semantic_roles"]
        self.domain = semantic_summary["detected_domain"]
        self.forecast_engine = forecast_engine
        self.context = ChatContext()

    def _resolve_entities(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolves target dimension and measure from query text or previous context."""
        q_lower = query.lower()
        matched_measure = None
        matched_dimension = None

        all_measures = self.semantic.get("all_measures", [])
        all_dimensions = self.semantic.get("all_dimensions", [])

        # Check measures in query
        for m in all_measures:
            m_clean = m.lower().replace('_', ' ')
            if m_clean in q_lower or any(word in q_lower for word in m_clean.split()):
                matched_measure = m
                break

        # Check domain synonym measures
        if not matched_measure:
            if any(k in q_lower for k in ["sales", "revenue", "turnover", "income"]):
                matched_measure = next((m for m in all_measures if any(k in m.lower() for k in ["sales", "rev", "amount"])), None)
            elif any(k in q_lower for k in ["profit", "margin", "gain", "earnings"]):
                matched_measure = next((m for m in all_measures if "profit" in m.lower()), None)
            elif any(k in q_lower for k in ["spend", "po value", "cost", "value", "procurement"]):
                matched_measure = next((m for m in all_measures if any(k in m.lower() for k in ["net_value", "po_value", "amount", "spend", "value"])), None)
            elif any(k in q_lower for k in ["quantity", "qty", "units", "volume"]):
                matched_measure = next((m for m in all_measures if any(k in m.lower() for k in ["qty", "quantity", "volume"])), None)

        # Check dimensions in query
        for d in all_dimensions:
            d_clean = d.lower().replace('_', ' ')
            if d_clean in q_lower or any(word in q_lower for word in d_clean.split() if len(word) > 3):
                matched_dimension = d
                break

        # Check dimension values directly (e.g. if user asks "how did TechSource perform?")
        if not matched_dimension:
            for d in all_dimensions:
                try:
                    unique_vals = [str(v).lower() for v in self.df[d].dropna().unique() if len(str(v)) > 2]
                    for uv in unique_vals:
                        if uv in q_lower:
                            matched_dimension = d
                            break
                    if matched_dimension:
                        break
                except Exception:
                    pass

        # Context continuity: If entity missing, fall back to conversational memory
        final_measure = matched_measure or self.context.last_measure or self.semantic.get("primary_measure")
        final_dimension = matched_dimension or self.context.last_dimension or self.semantic.get("primary_dimension")

        return final_dimension, final_measure

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main NLP conversation dispatcher."""
        q = query.strip()
        q_lower = q.lower()
        self.context.add_turn("user", q)

        # 1. VISUALIZATION REQUEST ("show me a chart", "plot this", "visualize")
        if any(w in q_lower for w in ["chart", "plot", "graph", "visualize", "show as chart"]):
            if self.context.last_chart_spec:
                msg = f"Here is the visualization for the previously discussed {self.context.last_measure or 'data'} breakdown:"
                resp = {
                    "answer": msg,
                    "chart": self.context.last_chart_spec,
                    "table_data": None
                }
                self.context.add_turn("assistant", msg)
                return resp

        # 2. FORECASTING / PREDICTION REQUEST
        if any(w in q_lower for w in ["forecast", "predict", "next 3 months", "future", "what will", "next quarter"]):
            dim, measure = self._resolve_entities(q)
            if self.forecast_engine and measure:
                fc = self.forecast_engine.run_forecast(target_measure=measure, horizon=3)
                if fc.get("is_suitable"):
                    answer = (
                        f"**Predictive Forecast for {measure.replace('_', ' ').title()}**:\n\n"
                        f"{fc.get('narrative')}\n\n"
                        f"• **Projected Average**: {fc['metrics']['forecast_mean']:,.2f}\n"
                        f"• **Historical Average**: {fc['metrics']['historical_mean']:,.2f}\n"
                        f"• **Model Reliability (MAPE)**: {fc['metrics']['mape']}%\n"
                    )
                    self.context.update("FORECAST", dim, measure, chart=fc.get("plotly_spec"))
                    self.context.add_turn("assistant", answer)
                    return {
                        "answer": answer,
                        "chart": fc.get("plotly_spec"),
                        "table_data": None
                    }
                else:
                    msg = f"Forecasting could not be reliably executed: {fc.get('reason')}"
                    self.context.add_turn("assistant", msg)
                    return {"answer": msg, "chart": None, "table_data": None}

        # 3. RANKING / TOP-N / HIGHEST / LOWEST
        dim, measure = self._resolve_entities(q)
        if not measure or not dim or measure not in self.df.columns or dim not in self.df.columns:
            msg = (
                "I couldn't identify the specific dimension or measure from your question or current context. "
                f"You can ask about measures like `{', '.join(self.semantic.get('all_measures', [])[:4])}` "
                f"or dimensions like `{', '.join(self.semantic.get('all_dimensions', [])[:4])}`."
            )
            self.context.add_turn("assistant", msg)
            return {"answer": msg, "chart": None, "table_data": None}

        # Check for Top-N number
        n_match = re.search(r'\btop\s*(\d+)\b', q_lower)
        top_k = int(n_match.group(1)) if n_match else (1 if any(w in q_lower for w in ["highest", "best", "most", "top"]) else None)
        is_lowest = any(w in q_lower for w in ["lowest", "worst", "least", "bottom"])

        grouped = self.df.groupby(dim)[measure].sum().reset_index()
        total_sum = grouped[measure].sum()

        if is_lowest:
            grouped = grouped.sort_values(measure, ascending=True).reset_index(drop=True)
            k = top_k or 1
            sample = grouped.head(k)
            first_row = sample.iloc[0]
            val = float(first_row[measure])
            entity = first_row[dim]
            share = round((val / total_sum * 100) if total_sum > 0 else 0, 1)

            answer = (
                f"**{entity}** has the lowest {measure.replace('_', ' ')} at **{val:,.2f}** "
                f"({share}% of total {measure.replace('_', ' ')})."
            )
            chart = self._create_chat_bar_chart(grouped.head(10), dim, measure, f"Lowest {dim} by {measure}")
            self.context.update("RANKING_LOW", dim, measure, result_df=grouped.head(10), chart=chart)
            self.context.add_turn("assistant", answer)
            return {"answer": answer, "chart": chart, "table_data": sample.to_dict(orient="records")}

        elif top_k:
            grouped = grouped.sort_values(measure, ascending=False).reset_index(drop=True)
            sample = grouped.head(top_k)
            first_row = sample.iloc[0]
            top_entity = first_row[dim]
            top_val = float(first_row[measure])
            share = round((top_val / total_sum * 100) if total_sum > 0 else 0, 1)

            if top_k == 1:
                answer = (
                    f"**{top_entity}** ranks #1 in {measure.replace('_', ' ')} with **{top_val:,.2f}**, "
                    f"accounting for **{share}%** of the total {measure.replace('_', ' ')} across all {dim.replace('_', ' ')}s."
                )
            else:
                top_items_text = ", ".join([f"**{r[dim]}** ({r[measure]:,.0f})" for _, r in sample.head(3).iterrows()])
                answer = (
                    f"The top {top_k} {dim.replace('_', ' ')}s by {measure.replace('_', ' ')} are led by {top_items_text}. "
                    f"Together, the top {top_k} generate {round((sample[measure].sum() / total_sum * 100), 1)}% of total volume."
                )

            chart = self._create_chat_bar_chart(grouped.head(max(top_k, 5)), dim, measure, f"Top {dim} by {measure}")
            self.context.update("RANKING_HIGH", dim, measure, result_df=sample, chart=chart)
            self.context.add_turn("assistant", answer)
            return {"answer": answer, "chart": chart, "table_data": sample.to_dict(orient="records")}

        # 4. TREND QUERY
        elif any(w in q_lower for w in ["trend", "over time", "monthly", "growth"]):
            time_dim = self.semantic.get("primary_time")
            if time_dim and time_dim in self.df.columns:
                t_df = self.df[[time_dim, measure]].dropna().copy()
                t_df["_dt"] = pd.to_datetime(t_df[time_dim], errors='coerce')
                t_df = t_df.dropna(subset=["_dt"]).sort_values("_dt")
                t_df["_period"] = t_df["_dt"].dt.to_period("M").astype(str)
                monthly = t_df.groupby("_period")[measure].sum().reset_index()

                start_val = monthly.iloc[0][measure]
                end_val = monthly.iloc[-1][measure]
                chg = round(((end_val - start_val) / start_val * 100) if start_val > 0 else 0, 1)
                dir_word = "growth" if chg >= 0 else "decline"

                answer = (
                    f"The {measure.replace('_', ' ')} trend spans from **{monthly.iloc[0]['_period']}** to **{monthly.iloc[-1]['_period']}**. "
                    f"Over this timeline, monthly volume shifted by **{chg:+0.1f}%** ({dir_word})."
                )

                chart = {
                    "data": [{
                        "x": monthly["_period"].tolist(),
                        "y": [round(float(v), 2) for v in monthly[measure].tolist()],
                        "type": "scatter",
                        "mode": "lines+markers",
                        "line": {"color": "#6366f1", "width": 3},
                        "marker": {"size": 6, "color": "#818cf8"}
                    }],
                    "layout": {
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)",
                        "font": {"color": "#94a3b8", "family": "Inter"},
                        "title": f"{measure.replace('_', ' ').title()} Monthly Trend",
                        "xaxis": {"gridcolor": "rgba(255,255,255,0.06)"},
                        "yaxis": {"gridcolor": "rgba(255,255,255,0.06)"}
                    }
                }
                self.context.update("TREND", time_dim, measure, result_df=monthly, chart=chart)
                self.context.add_turn("assistant", answer)
                return {"answer": answer, "chart": chart, "table_data": monthly.to_dict(orient="records")}

        # 5. GENERAL BREAKDOWN (e.g. "show sales by category")
        grouped = grouped.sort_values(measure, ascending=False).reset_index(drop=True)
        top_row = grouped.iloc[0]
        answer = (
            f"Breakdown of **{measure.replace('_', ' ')}** by **{dim.replace('_', ' ')}** "
            f"across {len(grouped)} segments. The leader is **{top_row[dim]}** with **{top_row[measure]:,.2f}** "
            f"({round(top_row[measure]/total_sum*100, 1)}% share)."
        )
        chart = self._create_chat_bar_chart(grouped.head(10), dim, measure, f"{measure} by {dim}")
        self.context.update("BREAKDOWN", dim, measure, result_df=grouped.head(10), chart=chart)
        self.context.add_turn("assistant", answer)
        return {"answer": answer, "chart": chart, "table_data": grouped.head(10).to_dict(orient="records")}

    def _create_chat_bar_chart(self, df_sub: pd.DataFrame, dim: str, measure: str, title: str) -> Dict[str, Any]:
        """Helper to generate responsive Plotly bar chart for chatbot answers."""
        return {
            "data": [{
                "x": [str(v) for v in df_sub[dim].tolist()],
                "y": [round(float(v), 2) for v in df_sub[measure].tolist()],
                "type": "bar",
                "marker": {"color": "#6366f1", "line": {"color": "#4f46e5", "width": 1}}
            }],
            "layout": {
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "font": {"color": "#94a3b8", "family": "Inter", "size": 11},
                "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
                "title": title.title(),
                "xaxis": {"gridcolor": "rgba(255,255,255,0.06)"},
                "yaxis": {"gridcolor": "rgba(255,255,255,0.06)"}
            }
        }
