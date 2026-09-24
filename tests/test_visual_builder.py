import pandas as pd
import pytest

from backend.core.visual_builder import VisualBuilder, VisualizationError


def test_build_grouped_visual():
    df = pd.DataFrame({
        "Region": ["South", "South", "North"],
        "Sales": [100, 200, 300],
    })
    result = VisualBuilder(df).build(
        chart_type="bar",
        dimension="Region",
        measure="Sales",
        aggregation="sum",
        top_n=2,
    )
    assert result["chart_type"] == "bar"
    assert result["data"][0]["value"] == 300


def test_build_filtered_visual():
    df = pd.DataFrame({
        "Region": ["South", "South", "North"],
        "Sales": [100, 200, 300],
    })
    result = VisualBuilder(df).build(
        chart_type="column",
        dimension="Region",
        measure="Sales",
        filters={"Region": ["South"]},
    )
    assert result["data"][0]["value"] == 300


def test_invalid_visual_configuration():
    df = pd.DataFrame({"Region": ["South"], "Sales": [100]})
    builder = VisualBuilder(df)

    with pytest.raises(VisualizationError):
        builder.build("bar", "Missing", "Sales")

    with pytest.raises(VisualizationError):
        builder.build("bar", "Region", "Sales", aggregation="unknown")

    with pytest.raises(VisualizationError):
        builder.build("bar", "Region", "Sales", top_n=0)
