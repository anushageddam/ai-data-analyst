import pandas as pd
import pytest

from backend.core.measure_engine import MeasureEngine, MeasureError


def sample_df():
    return pd.DataFrame(
        {
            "Sales": [100, 200, 300, 400],
            "Profit": [10, 40, 60, 80],
            "Region": ["South", "South", "North", "North"],
        }
    )


def test_basic_aggregations():
    engine = MeasureEngine(sample_df())

    assert engine.evaluate("SUM", "Sales") == 1000.0
    assert engine.evaluate("AVERAGE", "Sales") == 250.0
    assert engine.evaluate("MIN", "Sales") == 100.0
    assert engine.evaluate("MAX", "Sales") == 400.0
    assert engine.evaluate("MEDIAN", "Sales") == 250.0
    assert engine.evaluate("COUNT", "Region") == 4.0
    assert engine.evaluate("DISTINCTCOUNT", "Region") == 2.0


def test_ratio_and_filtered_calculation():
    engine = MeasureEngine(sample_df())

    assert engine.ratio("Profit", "Sales", as_percentage=True) == 19.0
    assert engine.filtered("SUM", "Sales", {"Region": "South"}) == 300.0


def test_validation_and_divide_by_zero():
    df = sample_df()

    assert MeasureEngine.validate_definition("SUM", "Sales", df)["valid"]
    assert not MeasureEngine.validate_definition("SUM", "Region", df)["valid"]
    assert engine_divide(df) == 0.0

    with pytest.raises(MeasureError):
        MeasureEngine(df).evaluate("SUM", "Missing")


def engine_divide(df):
    return MeasureEngine(df).divide(10, 0)


def test_custom_measure_creation_and_validation():
    from backend.core.measure_engine import CustomMeasureEngine

    engine = CustomMeasureEngine(sample_df())
    result = engine.create("Profit Margin", "Profit / Sales")

    assert result["name"] == "Profit Margin"
    assert result["formula"] == "Profit / Sales"
    assert result["value"] == 19.0
    assert engine.measures["Profit Margin"]["value"] == 19.0


def test_custom_measure_rejects_missing_field():
    from backend.core.measure_engine import CustomMeasureEngine

    with pytest.raises(MeasureError):
        CustomMeasureEngine(sample_df()).create("Bad", "Profit / Revenue")
