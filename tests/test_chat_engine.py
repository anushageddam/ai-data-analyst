import pandas as pd

from backend.core.chat_engine import AIChatEngine


def _semantic():
    return {
        "semantic_roles": {
            "all_measures": ["Sales", "Quantity"],
            "all_dimensions": ["Region", "Category"],
            "primary_measure": "Sales",
            "primary_dimension": "Region",
            "primary_time": "Date",
        },
        "detected_domain": "sales",
    }


def test_chat_engine_uses_planner_for_total():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        "Region": ["East", "West"],
        "Sales": [100, 250],
        "Quantity": [1, 2],
    })
    engine = AIChatEngine(df, _semantic())
    result = engine.process_query("What is the total sales?")

    assert result["analysis_plan"]["intent"] == "aggregation"
    assert result["analysis_plan"]["measure"] == "Sales"
    assert result["ai_analysis_plan"]["execution"]["deterministic"] is True
    assert result["ai_analysis_plan"]["validation"]["measure_exists"] is True
    assert result["answer"].endswith("350.00**")


def test_chat_engine_uses_planner_for_top_n():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "Region": ["East", "West", "East"],
        "Sales": [100, 300, 200],
        "Quantity": [1, 3, 2],
    })
    engine = AIChatEngine(df, _semantic())
    result = engine.process_query("top 2 regions by sales")

    assert result["analysis_plan"]["intent"] == "ranking"
    assert {row["Region"] for row in result["table_data"]} == {"East", "West"}
    assert all(row["Sales"] == 300 for row in result["table_data"])


def test_chat_engine_falls_back_for_non_analytical_message():
    df = pd.DataFrame({
        "Region": ["East"],
        "Sales": [100],
        "Quantity": [1],
    })
    engine = AIChatEngine(df, _semantic())
    result = engine.process_query("hello")

    assert "answer" in result
    assert "chart" in result
