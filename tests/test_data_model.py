import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.data_model import AnalyticalModel
from backend.core.data_sources import load_sample_dataset
from backend.core.profiler import DataProfiler
from backend.core.semantic import SemanticEngine


def test_analytical_model_builds_from_existing_pipeline():
    df, _ = load_sample_dataset("sap_procurement")

    profiler = DataProfiler(df)
    prof_out = profiler.profile_dataset()

    semantic = SemanticEngine(df, prof_out)
    sem_out = semantic.get_understanding_summary()

    model = AnalyticalModel(df, prof_out, sem_out).build()

    assert len(model["tables"]) == 1
    assert model["tables"][0]["row_count"] == len(df)
    assert model["tables"][0]["column_count"] == len(df.columns)
    assert len(model["tables"][0]["fields"]) == len(df.columns)


def test_analytical_model_field_roles_and_validation():
    df, _ = load_sample_dataset("enterprise_sales")

    profiler = DataProfiler(df)
    prof_out = profiler.profile_dataset()
    semantic = SemanticEngine(df, prof_out)
    sem_out = semantic.get_understanding_summary()

    model = AnalyticalModel(df, prof_out, sem_out).build()
    fields = model["tables"][0]["fields"]
    field_names = {field["name"] for field in fields}

    assert field_names == {str(column) for column in df.columns}
    assert all(field["role"] in {"Measure", "Date", "Identifier", "Dimension", "Other"} for field in fields)

    existing_field = next(iter(field_names))
    assert AnalyticalModel.validate_field(model, existing_field) is True
    assert AnalyticalModel.validate_field(model, "__missing_field__") is False
