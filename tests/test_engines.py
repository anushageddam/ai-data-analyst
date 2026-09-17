import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from backend.core.data_sources import load_sample_dataset
from backend.core.profiler import DataProfiler
from backend.core.semantic import SemanticEngine
from backend.core.kpi_engine import KPIEngine
from backend.core.vis_engine import VisDecisionEngine
from backend.core.insights import EvidenceInsightEngine
from backend.core.forecasting import ForecastEngine
from backend.core.chat_engine import AIChatEngine


def test_profiler_and_semantic_procurement():
    """Tests data profiling and semantic detection on SAP Procurement dataset."""
    df, meta = load_sample_dataset("sap_procurement")
    assert len(df) > 0

    profiler = DataProfiler(df)
    prof_out = profiler.profile_dataset()
    
    assert prof_out["summary"]["total_rows"] == len(df)
    assert prof_out["summary"]["data_health_score"] >= 80

    semantic = SemanticEngine(df, prof_out)
    sem_out = semantic.get_understanding_summary()

    # Must identify domain as Procurement / SAP ERP!
    assert "Procurement" in sem_out["detected_domain"]
    assert sem_out["primary_measure"] is not None
    assert sem_out["primary_dimension"] is not None


def test_profiler_and_semantic_sales():
    """Tests data profiling and semantic detection on Sales dataset."""
    df, meta = load_sample_dataset("enterprise_sales")
    assert len(df) > 0

    profiler = DataProfiler(df)
    prof_out = profiler.profile_dataset()
    
    semantic = SemanticEngine(df, prof_out)
    sem_out = semantic.get_understanding_summary()

    # Must identify domain as Sales & Commerce!
    assert "Sales" in sem_out["detected_domain"]
    assert "Sales" in sem_out["primary_measure"] or "sales" in sem_out["primary_measure"].lower()


def test_kpis_and_visuals():
    """Tests KPI generation and intelligent visual decision rules."""
    df, meta = load_sample_dataset("sap_procurement")
    profiler = DataProfiler(df)
    prof_out = profiler.profile_dataset()
    semantic = SemanticEngine(df, prof_out)
    sem_out = semantic.get_understanding_summary()

    kpi_eng = KPIEngine(df, sem_out)
    kpis = kpi_eng.generate_kpis()
    assert len(kpis) >= 3
    assert any("PO" in k["title"] or "Spend" in k["title"] or "Value" in k["title"] for k in kpis)

    vis_eng = VisDecisionEngine(df, sem_out, prof_out)
    visuals = vis_eng.generate_dashboard_visuals()
    assert len(visuals) >= 2
    # Ensure chart types are valid
    valid_types = {"line_area", "donut", "bar", "horizontal_bar", "grouped_bar", "scatter", "heatmap"}
    for v in visuals:
        assert v["chart_type"] in valid_types


def test_insights_and_forecasting():
    """Tests evidence-based insights and time-series forecast."""
    df, meta = load_sample_dataset("sap_procurement")
    profiler = DataProfiler(df)
    prof_out = profiler.profile_dataset()
    semantic = SemanticEngine(df, prof_out)
    sem_out = semantic.get_understanding_summary()

    insight_eng = EvidenceInsightEngine(df, sem_out, prof_out)
    insights = insight_eng.generate_insights()
    assert len(insights) >= 2
    for item in insights:
        assert "evidence" in item
        assert item["metric_value"] is not None

    forecast_eng = ForecastEngine(df, sem_out)
    fc = forecast_eng.run_forecast(horizon=3)
    assert fc["is_suitable"] is True
    assert "forecast_mean" in fc["metrics"]
    assert len(fc["plotly_spec"]["data"]) >= 3


def test_ai_chat_context_and_charts():
    """Tests conversational AI query engine, continuity, and chart generation."""
    df, meta = load_sample_dataset("sap_procurement")
    profiler = DataProfiler(df)
    prof_out = profiler.profile_dataset()
    semantic = SemanticEngine(df, prof_out)
    sem_out = semantic.get_understanding_summary()
    forecast_eng = ForecastEngine(df, sem_out)

    chat = AIChatEngine(df, sem_out, forecast_eng)

    # 1. Ranking query
    resp1 = chat.process_query("Which vendor has the highest spend?")
    assert "Global Steel Co" in resp1["answer"] or "highest" in resp1["answer"]
    assert resp1["chart"] is not None

    # 2. Context continuity: "Show as a chart"
    resp2 = chat.process_query("Show me a chart")
    assert resp2["chart"] is not None

    # 3. Forecast query
    resp3 = chat.process_query("What will spend look like next 3 months?")
    assert "Predictive Forecast" in resp3["answer"] or "projected" in resp3["answer"]
    print("All engine tests passed successfully!")


if __name__ == "__main__":
    test_profiler_and_semantic_procurement()
    print("[PASS] test_profiler_and_semantic_procurement")
    test_profiler_and_semantic_sales()
    print("[PASS] test_profiler_and_semantic_sales")
    test_kpis_and_visuals()
    print("[PASS] test_kpis_and_visuals")
    test_insights_and_forecasting()
    print("[PASS] test_insights_and_forecasting")
    test_ai_chat_context_and_charts()
    print("[PASS] test_ai_chat_context_and_charts")
    print("\n>>> ALL 5 TEST SUITES PASSED VERIFICATION! <<<")
