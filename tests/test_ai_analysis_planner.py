import pandas as pd
import pytest

from backend.core.ai_analysis_planner import AIAnalysisPlanner
from backend.core.analytics_planner import AnalyticsPlanError


def _semantic():
    return {
        "detected_domain": "Sales & Commerce",
        "semantic_roles": {
            "all_measures": ["Sales", "Quantity"],
            "all_dimensions": ["Region", "Category"],
            "primary_measure": "Sales",
            "primary_dimension": "Region",
            "primary_time": "Date",
        },
    }


@pytest.fixture
def df():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01"]),
        "Region": ["East", "West", "East"],
        "Category": ["A", "A", "B"],
        "Sales": [100, 200, 300],
        "Quantity": [1, 2, 3],
    })


def test_ai_planner_builds_structured_plan(df):
    planner = AIAnalysisPlanner(df, _semantic())
    result = planner.analyze("What is the total sales?")

    plan = result["plan"]
    assert result["validated"] is True
    assert plan["intent"] == "aggregation"
    assert plan["required_fields"]["measure"] == "Sales"
    assert plan["calculation"]["aggregation"] == "SUM"
    assert plan["validation"]["measure_exists"] is True
    assert plan["execution"]["deterministic"] is True
    assert result["result"]["value"] == 600.0


def test_ai_planner_accepts_only_existing_filter_values(df):
    planner = AIAnalysisPlanner(df, _semantic())

    result = planner.analyze("show sales by region where region is East")
    assert result["plan"]["context"]["active_filters"] == [
        {"field": "Region", "operator": "equals", "value": "East"}
    ]


def test_ai_planner_does_not_invent_filter_values(df):
    planner = AIAnalysisPlanner(df, _semantic())

    result = planner.plan("show sales by region where region is Mars")
    assert result["plan"]["context"]["active_filters"] == []


def test_ai_planner_rejects_empty_question(df):
    with pytest.raises(AnalyticsPlanError):
        AIAnalysisPlanner(df, _semantic()).plan("")
