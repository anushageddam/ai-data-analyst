import pandas as pd
import pytest

from backend.core.analytics_planner import AnalyticsPlanner, AnalyticsPlanError


def _semantic():
    return {
        "semantic_roles": {
            "all_measures": ["Sales", "Quantity"],
            "all_dimensions": ["Region", "Category"],
            "primary_measure": "Sales",
            "primary_dimension": "Region",
            "primary_time": "Date",
        }
    }


@pytest.fixture
def df():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-20"]),
        "Region": ["East", "West", "East", "West"],
        "Category": ["A", "A", "B", "B"],
        "Sales": [100, 200, 300, 400],
        "Quantity": [1, 2, 3, 4],
    })


def test_plan_and_execute_total(df):
    result = AnalyticsPlanner(df, _semantic()).analyze("What is the total sales?")
    assert result["validated"] is True
    assert result["plan"]["intent"] == "aggregation"
    assert result["result"]["value"] == 1000.0


def test_ranking_uses_real_grouped_values(df):
    result = AnalyticsPlanner(df, _semantic()).analyze("top 2 regions by sales")
    rows = result["result"]["data"]
    assert [row["Region"] for row in rows] == ["West", "East"]
    assert [row["Sales"] for row in rows] == [600, 400]


def test_trend_uses_detected_date(df):
    result = AnalyticsPlanner(df, _semantic()).analyze("show sales trend over time")
    rows = result["result"]["data"]
    assert rows == [{"_period": "2026-01", "Sales": 300}, {"_period": "2026-02", "Sales": 700}]


def test_invalid_question_is_rejected(df):
    with pytest.raises(AnalyticsPlanError):
        AnalyticsPlanner(df, _semantic()).analyze("")


def test_missing_measure_is_rejected(df):
    semantic = _semantic()
    semantic["semantic_roles"]["primary_measure"] = "Profit"
    semantic["semantic_roles"]["all_measures"] = ["Profit"]
    with pytest.raises(AnalyticsPlanError):
        AnalyticsPlanner(df, semantic).analyze("total profit")
