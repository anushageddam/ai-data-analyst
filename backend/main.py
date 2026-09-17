import os
import io
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel

def sanitize_for_json(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    elif pd.isna(obj):
        return None
    return obj

from backend.core.data_sources import FileDataSource, load_sample_dataset
from backend.core.profiler import DataProfiler
from backend.core.semantic import SemanticEngine
from backend.core.kpi_engine import KPIEngine
from backend.core.vis_engine import VisDecisionEngine
from backend.core.insights import EvidenceInsightEngine
from backend.core.forecasting import ForecastEngine
from backend.core.chat_engine import AIChatEngine

app = FastAPI(title="YOO PROJECT - AI BI & Analytics Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
class SessionStore:
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.profiler_output: Dict[str, Any] = {}
        self.semantic_output: Dict[str, Any] = {}
        self.kpis: List[Dict[str, Any]] = []
        self.visuals: List[Dict[str, Any]] = []
        self.insights: List[Dict[str, Any]] = []
        self.forecast_engine: Optional[ForecastEngine] = None
        self.chat_engine: Optional[AIChatEngine] = None
        self.staged_excel_bytes: Optional[bytes] = None
        self.staged_filename: Optional[str] = None

    def initialize_from_df(self, df: pd.DataFrame, meta: Dict[str, Any]):
        self.raw_df = df.copy()
        self.filtered_df = df.copy()
        self.metadata = meta

        # 1. Profile Data
        profiler = DataProfiler(self.filtered_df)
        self.profiler_output = profiler.profile_dataset()

        # 2. Understand Semantics & Business Domain
        semantic_engine = SemanticEngine(self.filtered_df, self.profiler_output)
        self.semantic_output = semantic_engine.get_understanding_summary()

        # 3. Generate KPIs
        kpi_engine = KPIEngine(self.filtered_df, self.semantic_output)
        self.kpis = kpi_engine.generate_kpis()

        # 4. Generate Visuals
        vis_engine = VisDecisionEngine(self.filtered_df, self.semantic_output, self.profiler_output)
        self.visuals = vis_engine.generate_dashboard_visuals()

        # 5. Generate Evidence-Based Insights
        insight_engine = EvidenceInsightEngine(self.filtered_df, self.semantic_output, self.profiler_output)
        self.insights = insight_engine.generate_insights()

        # 6. Initialize Forecasting & Chat Engines
        self.forecast_engine = ForecastEngine(self.filtered_df, self.semantic_output)
        self.chat_engine = AIChatEngine(self.filtered_df, self.semantic_output, self.forecast_engine)

    def apply_filters(self, filters: Dict[str, List[Any]]):
        """Filters the raw DataFrame and recomputes all metrics."""
        if self.raw_df is None:
            return

        df = self.raw_df.copy()
        for col, values in filters.items():
            if col in df.columns and values:
                # Handle string conversion for robust matching
                str_vals = [str(v) for v in values]
                df = df[df[col].astype(str).isin(str_vals)]

        self.filtered_df = df

        # Re-run KPI & Visual & Insight engines over filtered subset
        kpi_engine = KPIEngine(self.filtered_df, self.semantic_output)
        self.kpis = kpi_engine.generate_kpis()

        vis_engine = VisDecisionEngine(self.filtered_df, self.semantic_output, self.profiler_output)
        self.visuals = vis_engine.generate_dashboard_visuals()

        insight_engine = EvidenceInsightEngine(self.filtered_df, self.semantic_output, self.profiler_output)
        self.insights = insight_engine.generate_insights()

        self.forecast_engine = ForecastEngine(self.filtered_df, self.semantic_output)
        self.chat_engine = AIChatEngine(self.filtered_df, self.semantic_output, self.forecast_engine)


session = SessionStore()


# --- REQUEST SCHEMAS ---
class FilterRequest(BaseModel):
    filters: Dict[str, List[Any]] = {}

class SheetSelectRequest(BaseModel):
    sheet_name: str

class SampleLoadRequest(BaseModel):
    sample_id: str

class ForecastRequest(BaseModel):
    measure: Optional[str] = None
    horizon: int = 3

class ChatRequest(BaseModel):
    query: str


# --- API ENDPOINTS ---

@app.get("/api/status")
def get_status():
    if session.raw_df is None:
        return JSONResponse(content={"loaded": False})
    return JSONResponse(content=sanitize_for_json({
        "loaded": True,
        "metadata": session.metadata,
        "rows": len(session.filtered_df) if session.filtered_df is not None else 0,
        "total_rows": len(session.raw_df),
        "columns": len(session.raw_df.columns),
        "domain": session.semantic_output.get("detected_domain"),
        "health_score": session.profiler_output.get("summary", {}).get("data_health_score"),
        "health_status": session.profiler_output.get("summary", {}).get("data_health_status"),
        "filter_dimensions": session.semantic_output.get("filter_dimensions", [])
    }))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_data.csv"
    contents = await file.read()
    
    is_excel = filename.lower().endswith(('.xlsx', '.xls'))
    if is_excel:
        source = FileDataSource(contents, filename)
        sheets = source.get_sheets()
        session.staged_excel_bytes = contents
        session.staged_filename = filename

        if len(sheets) > 1:
            return {
                "status": "multiple_sheets",
                "filename": filename,
                "sheets": sheets,
                "message": "Multiple sheets detected. Please choose a sheet to load."
            }
        else:
            df = source.load(sheets[0] if sheets else None)
            meta = source.get_metadata()
            session.initialize_from_df(df, meta)
            return {"status": "success", "metadata": meta}
    else:
        source = FileDataSource(contents, filename)
        df = source.load()
        meta = source.get_metadata()
        session.initialize_from_df(df, meta)
        return {"status": "success", "metadata": meta}


@app.post("/api/select-sheet")
def select_sheet(req: SheetSelectRequest):
    if not session.staged_excel_bytes or not session.staged_filename:
        raise HTTPException(status_code=400, detail="No staged Excel workbook found.")
    
    source = FileDataSource(session.staged_excel_bytes, session.staged_filename, sheet_name=req.sheet_name)
    df = source.load(sheet_name=req.sheet_name)
    meta = source.get_metadata()
    session.initialize_from_df(df, meta)
    return {"status": "success", "metadata": meta}


@app.post("/api/load-sample")
def load_sample(req: SampleLoadRequest):
    try:
        df, meta = load_sample_dataset(req.sample_id)
        session.initialize_from_df(df, meta)
        return {"status": "success", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dashboard")
def get_dashboard(req: Optional[FilterRequest] = None):
    if session.raw_df is None:
        # Auto-load SAP Procurement sample if nothing loaded yet
        df, meta = load_sample_dataset("sap_procurement")
        session.initialize_from_df(df, meta)

    if req and req.filters:
        session.apply_filters(req.filters)

    # Prepare available filter values for the UI dropdowns
    filter_options = {}
    for fdim in session.semantic_output.get("filter_dimensions", []):
        if fdim in session.raw_df.columns:
            vals = sorted([str(v) for v in session.raw_df[fdim].dropna().unique().tolist()])
            filter_options[fdim] = vals

    return JSONResponse(content=sanitize_for_json({
        "metadata": session.metadata,
        "active_rows": len(session.filtered_df),
        "total_rows": len(session.raw_df),
        "semantic_summary": session.semantic_output,
        "kpis": session.kpis,
        "visuals": session.visuals,
        "insights": session.insights,
        "profiler_summary": session.profiler_output.get("summary", {}),
        "filter_options": filter_options
    }))


@app.get("/api/profiler")
def get_profiler_details():
    if session.raw_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    return JSONResponse(content=sanitize_for_json(session.profiler_output))


@app.post("/api/forecast")
def run_forecast(req: ForecastRequest):
    if session.forecast_engine is None:
        raise HTTPException(status_code=400, detail="No dataset loaded for forecasting.")
    result = session.forecast_engine.run_forecast(target_measure=req.measure, horizon=req.horizon)
    return JSONResponse(content=sanitize_for_json(result))


@app.post("/api/chat")
def chat_query(req: ChatRequest):
    if session.chat_engine is None:
        raise HTTPException(status_code=400, detail="No dataset loaded for conversation.")
    resp = session.chat_engine.process_query(req.query)
    return JSONResponse(content=sanitize_for_json(resp))


@app.get("/api/explorer")
def get_explorer_data(page: int = 1, page_size: int = 25, search: str = "", sort_col: str = "", sort_dir: str = "asc"):
    if session.filtered_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    df = session.filtered_df.copy()

    # Search filter
    if search.strip():
        q = search.lower()
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(q, na=False)).any(axis=1)
        df = df[mask]

    # Sort
    if sort_col and sort_col in df.columns:
        ascending = (sort_dir.lower() != "desc")
        df = df.sort_values(sort_col, ascending=ascending)

    total_records = len(df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    sliced = df.iloc[start_idx:end_idx]

    return JSONResponse(content=sanitize_for_json({
        "total_records": total_records,
        "page": page,
        "page_size": page_size,
        "columns": list(df.columns),
        "data": sliced.fillna("").to_dict(orient="records")
    }))


@app.get("/api/export")
def export_csv():
    if session.filtered_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    stream = io.StringIO()
    session.filtered_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    filename = f"yoo_export_{session.metadata.get('filename', 'data.csv')}"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


# Mount frontend static assets
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>YOO PROJECT Frontend initializing...</h1>"
